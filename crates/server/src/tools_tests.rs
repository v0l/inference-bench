use super::*;
use serde_json::json;

fn weather_tools() -> Vec<Value> {
    vec![
        json!({"type": "function", "function": {"name": "get_current_weather", "description": "Get the current weather", "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name"},
                "state": {"type": "string", "description": "The state code"},
                "unit": {"type": "string", "enum": ["fahrenheit", "celsius"]},
            },
            "required": ["city", "state"],
        }}}),
        json!({"type": "function", "function": {"name": "calculate_area", "description": "Calculate area of a shape", "parameters": {
            "type": "object",
            "properties": {"shape": {"type": "string"}, "dimensions": {"type": "object"}, "precision": {"type": "integer"}},
        }}}),
    ]
}

fn one_tool(name: &str, properties: Value) -> Vec<Value> {
    vec![json!({"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": properties}}})]
}

const DALLAS: &str = "<tool_call>\n<function=get_current_weather>\n<parameter=city>\nDallas\n</parameter>\n<parameter=state>\nTX\n</parameter>\n<parameter=unit>\nfahrenheit\n</parameter>\n</function>\n</tool_call>";
const DALLAS_ARGS: &str = r#"{"city": "Dallas", "state": "TX", "unit": "fahrenheit"}"#;

fn calls(p: &Parsed) -> Vec<(&str, &str)> {
    p.tool_calls.iter().map(|c| (c.name.as_str(), c.arguments.as_str())).collect()
}

#[derive(Default, Debug, PartialEq)]
struct Streamed {
    content: String,
    reasoning: String,
    calls: Vec<(String, String, String)>,
}

fn stream(tools: &[Value], reasoning: bool, chunks: &[&str]) -> Streamed {
    let mut p = OutputParser::new(tools, true, reasoning);
    let mut out = Streamed::default();
    let mut deltas = Vec::new();
    for c in chunks {
        deltas.extend(p.feed(c));
    }
    deltas.extend(p.finish());
    for d in deltas {
        match d {
            Delta::Content(t) => out.content.push_str(&t),
            Delta::Reasoning(t) => out.reasoning.push_str(&t),
            Delta::Call { index, id, name } => {
                assert_eq!(index, out.calls.len());
                assert!(id.starts_with("chatcmpl-tool-"));
                out.calls.push((id, name, String::new()));
            }
            Delta::Args { index, arguments } => out.calls[index].2.push_str(&arguments),
        }
    }
    out
}

fn by_char(text: &str) -> Vec<&str> {
    let mut v = Vec::new();
    let mut start = 0;
    for (i, _) in text.char_indices().skip(1) {
        v.push(&text[start..i]);
        start = i;
    }
    v.push(&text[start..]);
    v
}

fn assert_stream_matches(tools: &[Value], text: &str) {
    let whole = OutputParser::parse(tools, true, false, text);
    for chunks in [vec![text], by_char(text)] {
        let s = stream(tools, false, &chunks);
        let got: Vec<(&str, &str)> = s.calls.iter().map(|(_, n, a)| (n.as_str(), a.as_str())).collect();
        assert_eq!(got, calls(&whole), "{chunks:?}");
        assert_eq!(s.content.trim(), whole.content.as_deref().unwrap_or(""));
    }
}

#[test]
fn text_without_tool_calls_is_all_content() {
    let p = OutputParser::parse(&weather_tools(), true, false, "This is a test response without any tool calls");
    assert!(p.tool_calls.is_empty());
    assert_eq!(p.content.as_deref(), Some("This is a test response without any tool calls"));
}

#[test]
fn a_single_tool_call_parses_to_vllm_arguments() {
    let p = OutputParser::parse(&weather_tools(), true, false, DALLAS);
    assert_eq!(calls(&p), [("get_current_weather", DALLAS_ARGS)]);
    assert_eq!(p.content, None);
}

#[test]
fn text_before_the_call_is_content() {
    let text = format!("Sure! Let me check the weather for you.{DALLAS}");
    let p = OutputParser::parse(&weather_tools(), true, false, &text);
    assert_eq!(calls(&p), [("get_current_weather", DALLAS_ARGS)]);
    assert_eq!(p.content.as_deref(), Some("Sure! Let me check the weather for you."));
    assert_stream_matches(&weather_tools(), &text);
}

#[test]
fn a_multiline_json_parameter_is_typed_by_the_schema() {
    let text = "<tool_call>\n<function=calculate_area>\n<parameter=shape>\nrectangle\n</parameter>\n<parameter=dimensions>\n{\"width\": 10, \n \"height\": 20}\n</parameter>\n<parameter=precision>\n2\n</parameter>\n</function>\n</tool_call>";
    let p = OutputParser::parse(&weather_tools(), true, false, text);
    assert_eq!(calls(&p), [("calculate_area", r#"{"shape": "rectangle", "dimensions": {"width": 10, "height": 20}, "precision": 2}"#)]);
    assert_stream_matches(&weather_tools(), text);
}

#[test]
fn parallel_calls_each_get_an_index() {
    let orlando = DALLAS.replace("Dallas", "Orlando").replace("TX", "FL").replace("fahrenheit", "celsius");
    let text = format!("{DALLAS}\n{orlando}");
    let p = OutputParser::parse(&weather_tools(), true, false, &text);
    assert_eq!(
        calls(&p),
        [("get_current_weather", DALLAS_ARGS), ("get_current_weather", r#"{"city": "Orlando", "state": "FL", "unit": "celsius"}"#)]
    );
    assert_eq!(p.content, None);
    assert_stream_matches(&weather_tools(), &text);
    let s = stream(&weather_tools(), false, &by_char(&text));
    assert_ne!(s.calls[0].0, s.calls[1].0);
    assert_eq!(s.content, "");
}

#[test]
fn typed_params_with_a_float_in_an_object() {
    let text = "Let me calculate that area for you.<tool_call>\n<function=calculate_area>\n<parameter=shape>\ncircle\n</parameter>\n<parameter=dimensions>\n{\"radius\": 15.5}\n</parameter>\n<parameter=precision>\n3\n</parameter>\n</function>\n</tool_call>";
    let p = OutputParser::parse(&weather_tools(), true, false, text);
    assert_eq!(calls(&p), [("calculate_area", r#"{"shape": "circle", "dimensions": {"radius": 15.5}, "precision": 3}"#)]);
    assert_eq!(p.content.as_deref(), Some("Let me calculate that area for you."));
}

#[test]
fn a_call_without_the_tool_call_tag_still_parses() {
    let text = "<function=get_current_weather>\n<parameter=city>\nDallas\n</parameter>\n<parameter=state>\nTX\n</parameter>\n</function>";
    let p = OutputParser::parse(&weather_tools(), true, false, text);
    assert_eq!(calls(&p), [("get_current_weather", r#"{"city": "Dallas", "state": "TX"}"#)]);
    let text = format!("I'll check the weather for you.\n\n{text}\n</tool_call>");
    let s = stream(&weather_tools(), false, &by_char(&text));
    assert_eq!(s.calls.len(), 1);
    assert!(s.content.contains("I'll check the weather for you."));
}

#[test]
fn parameters_are_typed_by_the_schema() {
    let tools = one_tool(
        "test_types",
        json!({"int_param": {"type": "integer"}, "float_param": {"type": "float"}, "bool_param": {"type": "boolean"}, "str_param": {"type": "string"}, "obj_param": {"type": "object"}}),
    );
    let text = "<tool_call>\n<function=test_types>\n<parameter=int_param>\n42\n</parameter>\n<parameter=float_param>\n3.14\n</parameter>\n<parameter=bool_param>\ntrue\n</parameter>\n<parameter=str_param>\nhello world\n</parameter>\n<parameter=obj_param>\n{\"key\": \"value\"}\n</parameter>\n</function>\n</tool_call>";
    let p = OutputParser::parse(&tools, true, false, text);
    assert_eq!(
        p.tool_calls[0].arguments,
        r#"{"int_param": 42, "float_param": 3.14, "bool_param": true, "str_param": "hello world", "obj_param": {"key": "value"}}"#
    );
}

#[test]
fn any_of_nullable_schemas_pick_the_non_null_type() {
    let tools = one_tool(
        "test_anyof",
        json!({
            "anyof_int": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": 5},
            "anyof_str": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "anyof_array": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]},
            "anyof_obj": {"anyOf": [{"type": "object"}, {"type": "null"}]},
            "type_as_array": {"type": ["integer", "null"]},
            "multi_non_null": {"anyOf": [{"type": "string"}, {"type": "integer"}, {"type": "null"}]},
        }),
    );
    let text = "<tool_call>\n<function=test_anyof>\n<parameter=anyof_int>\n5\n</parameter>\n<parameter=anyof_str>\nhello\n</parameter>\n<parameter=anyof_array>\n[\"a\", \"b\", \"c\"]\n</parameter>\n<parameter=anyof_obj>\n{\"key\": \"value\"}\n</parameter>\n<parameter=type_as_array>\n42\n</parameter>\n<parameter=multi_non_null>\nsome text\n</parameter>\n</function>\n</tool_call>";
    let p = OutputParser::parse(&tools, true, false, text);
    assert_eq!(
        p.tool_calls[0].arguments,
        r#"{"anyof_int": 5, "anyof_str": "hello", "anyof_array": ["a", "b", "c"], "anyof_obj": {"key": "value"}, "type_as_array": 42, "multi_non_null": "some text"}"#
    );
    let search = one_tool(
        "search_web",
        json!({"query": {"anyOf": [{"type": "string"}, {"type": "null"}]}, "count": {"anyOf": [{"type": "integer"}, {"type": "null"}]}, "verbose": {"anyOf": [{"type": "boolean"}, {"type": "null"}]}}),
    );
    let text = "<tool_call>\n<function=search_web>\n<parameter=query>\nvllm tool parser\n</parameter>\n<parameter=count>\n10\n</parameter>\n<parameter=verbose>\ntrue\n</parameter>\n</function>\n</tool_call>";
    let s = stream(&search, false, &by_char(text));
    assert_eq!(s.calls[0].2, r#"{"query": "vllm tool parser", "count": 10, "verbose": true}"#);
}

#[test]
fn a_missing_closing_parameter_tag_ends_at_the_next_parameter() {
    let text = "Let me check the weather for you:\n<tool_call>\n<function=get_current_weather>\n<parameter=city>\nDallas\n<parameter=state>\nTX\n</parameter>\n<parameter=unit>\nfahrenheit\n</parameter>\n</function>\n</tool_call>";
    let p = OutputParser::parse(&weather_tools(), true, false, text);
    assert_eq!(calls(&p), [("get_current_weather", DALLAS_ARGS)]);
    assert_eq!(p.content.as_deref(), Some("Let me check the weather for you:"));
    let s = stream(&weather_tools(), false, &by_char(text));
    assert_eq!(s.calls[0].2, DALLAS_ARGS);
}

#[test]
fn malformed_names_without_a_closing_bracket_do_not_crash() {
    let text = "<tool_call>\n<function=bad_func_no_gt\n</function>\n</tool_call>\n<tool_call>\n<function=get_current_weather>\n<parameter=city>Dallas</parameter>\n<parameter=state>TX</parameter>\n</function>\n</tool_call>";
    let p = OutputParser::parse(&weather_tools(), true, false, text);
    let valid: Vec<_> = p.tool_calls.iter().filter(|c| c.name == "get_current_weather").collect();
    assert_eq!(valid.len(), 1);
    assert_eq!(valid[0].arguments, r#"{"city": "Dallas", "state": "TX"}"#);
    OutputParser::parse(&weather_tools(), true, false, "<tool_call>\n<function=get_current_weather\n<parameter=city>Dallas</parameter>\n</function>\n</tool_call>");
}

#[test]
fn an_any_of_object_is_not_double_encoded() {
    let tools = one_tool("update_record", json!({"data": {"anyOf": [{"type": "object"}, {"type": "null"}]}}));
    let text = "<tool_call>\n<function=update_record>\n<parameter=data>{\"key\": \"value\", \"count\": 42}</parameter>\n</function>\n</tool_call>";
    assert_eq!(OutputParser::parse(&tools, true, false, text).tool_calls[0].arguments, r#"{"data": {"key": "value", "count": 42}}"#);
}

#[test]
fn speculative_chunks_with_several_parameters_parse() {
    let chunks = [
        "<tool_call>",
        "\n<function=get_current_weather>",
        "\n",
        "<parameter=city>\nDallas\n</parameter>\n<parameter=state>\nTX\n</parameter>\n<parameter=unit>\nfahrenheit\n</parameter>",
        "\n</function>",
        "\n</tool_call>",
    ];
    let s = stream(&weather_tools(), false, &chunks);
    assert_eq!(s.calls.len(), 1);
    assert_eq!(s.calls[0].2, DALLAS_ARGS);
}

#[test]
fn the_next_call_may_start_in_the_closing_chunk() {
    let chunks = [
        "<tool_call>\n",
        "<function=get_current_weather>\n",
        "<parameter=city>\nDallas\n</parameter>\n",
        "<parameter=state>\nTX\n</parameter>\n",
        "</function>",
        "\n</tool_call>\n<tool_call>\n<function=get_current_weather>\n<parameter=city>\nOrlando\n</parameter>\n<parameter=state>\nFL\n</parameter>\n</function>\n</tool_call>",
    ];
    let s = stream(&weather_tools(), false, &chunks);
    let args: Vec<&str> = s.calls.iter().map(|c| c.2.as_str()).collect();
    assert_eq!(args, [r#"{"city": "Dallas", "state": "TX"}"#, r#"{"city": "Orlando", "state": "FL"}"#]);
}

#[test]
fn string_arguments_are_not_double_serialized() {
    let tools = one_tool("greet", json!({"message": {"type": "string"}}));
    let p = OutputParser::parse(&tools, true, false, "<tool_call>\n<function=greet>\n<parameter=message>hello world</parameter>\n</function>\n</tool_call>");
    assert_eq!(p.tool_calls[0].arguments, r#"{"message": "hello world"}"#);
}

#[test]
fn only_the_wrapping_newlines_of_a_value_are_trimmed() {
    let tools = one_tool("write_file", json!({"content": {"type": "string"}}));
    let text = "<tool_call>\n<function=write_file>\n<parameter=content>\n    def foo():\n        return 1\n\n</parameter>\n</function>\n</tool_call>";
    let want = r#"{"content": "    def foo():\n        return 1\n"}"#;
    assert_eq!(OutputParser::parse(&tools, true, false, text).tool_calls[0].arguments, want);
    assert_eq!(stream(&tools, false, &by_char(text)).calls[0].2, want);
}

#[test]
fn without_a_schema_values_stay_strings() {
    let p = OutputParser::parse(&[], true, false, "<tool_call>\n<function=f>\n<parameter=n>\n7\n</parameter>\n</function>\n</tool_call>");
    assert_eq!(calls(&p), [("f", r#"{"n": "7"}"#)]);
}

#[test]
fn reasoning_splits_at_the_closing_think_tag() {
    let text = "Let me think.\nOK.\n</think>\n\nThe answer is 4.";
    let p = OutputParser::parse(&[], true, true, text);
    assert_eq!(p.reasoning.as_deref(), Some("Let me think.\nOK.\n"));
    assert_eq!(p.content.as_deref(), Some("\n\nThe answer is 4."));
    let s = stream(&[], true, &by_char(text));
    assert_eq!((s.reasoning.as_str(), s.content.as_str()), ("Let me think.\nOK.\n", "\n\nThe answer is 4."));
}

#[test]
fn a_tool_call_ends_reasoning_without_a_closing_tag() {
    let text = format!("<think>I should call it.\n{DALLAS}");
    let p = OutputParser::parse(&weather_tools(), true, true, &text);
    assert_eq!(p.reasoning.as_deref(), Some("I should call it.\n"));
    assert_eq!(calls(&p), [("get_current_weather", DALLAS_ARGS)]);
}

#[test]
fn reasoning_that_never_closes_is_all_reasoning() {
    let p = OutputParser::parse(&[], true, true, "still thinking about <it");
    assert_eq!(p.reasoning.as_deref(), Some("still thinking about <it"));
    assert_eq!(p.content, None);
}

#[test]
fn with_tool_parsing_off_tags_are_content() {
    let p = OutputParser::parse(&weather_tools(), false, false, DALLAS);
    assert!(p.tool_calls.is_empty());
    assert_eq!(p.content.as_deref(), Some(DALLAS));
}

#[test]
fn a_partial_tag_at_a_chunk_edge_is_held_back() {
    let mut p = OutputParser::new(&weather_tools(), true, false);
    assert_eq!(p.feed("Hello <tool"), vec![Delta::Content("Hello ".into())]);
    assert_eq!(p.feed("s are fun"), vec![Delta::Content("<tools are fun".into())]);
}
