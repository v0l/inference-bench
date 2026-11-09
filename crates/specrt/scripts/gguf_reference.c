#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "llama.h"

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: gguf_reference model.gguf out_prefix n_gpu_layers n_gen tok0 tok1 ...\n");
        return 2;
    }
    const char* model_path = argv[1];
    const char* prefix = argv[2];
    int n_gpu = atoi(argv[3]);
    int n_gen = atoi(argv[4]);
    int n_prompt = argc - 5;
    llama_token* prompt = malloc(sizeof(llama_token) * n_prompt);
    for (int i = 0; i < n_prompt; i++) prompt[i] = atoi(argv[5 + i]);

    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu;
    struct llama_model* model = llama_model_load_from_file(model_path, mp);
    if (!model) return 1;
    const struct llama_vocab* vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_prompt + n_gen + 8;
    cp.n_batch = n_prompt + n_gen + 8;
    cp.n_ubatch = cp.n_batch;
    struct llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;

    struct llama_batch batch = llama_batch_init(n_prompt, 0, 1);
    batch.n_tokens = n_prompt;
    for (int i = 0; i < n_prompt; i++) {
        batch.token[i] = prompt[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    if (llama_decode(ctx, batch)) return 1;

    char path[4096];
    snprintf(path, sizeof path, "%s.logits.bin", prefix);
    FILE* f = fopen(path, "wb");
    for (int i = 0; i < n_prompt; i++) fwrite(llama_get_logits_ith(ctx, i), sizeof(float), n_vocab, f);
    fclose(f);

    snprintf(path, sizeof path, "%s.greedy.txt", prefix);
    f = fopen(path, "w");
    int pos = n_prompt;
    const float* logits = llama_get_logits_ith(ctx, n_prompt - 1);
    for (int g = 0; g < n_gen; g++) {
        int best = 0;
        for (int v = 1; v < n_vocab; v++) if (logits[v] > logits[best]) best = v;
        fprintf(f, "%d\n", best);
        struct llama_batch one = llama_batch_get_one((llama_token*)&best, 1);
        one.pos = NULL;
        if (llama_decode(ctx, one)) return 1;
        logits = llama_get_logits_ith(ctx, 0);
        pos++;
    }
    fclose(f);
    printf("vocab %d prompt %d\n", n_vocab, n_prompt);
    return 0;
}
