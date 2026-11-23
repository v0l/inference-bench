try:
    from transformers import video_utils as _vu
    _init = _vu.VideoMetadata.__init__
    def _patched(self, *a, do_sample_frames=None, **k):
        _init(self, *a, **k)
    _vu.VideoMetadata.__init__ = _patched
except Exception:
    pass
try:
    from transformers.processing_utils import ProcessorMixin as _PM
    _validate = _PM.validate_inputs
    def _lenient(self, *a, **k):
        try:
            return _validate(self, *a, **k)
        except ValueError as e:
            if "no videos were passed" in str(e):
                return None
            raise
    _PM.validate_inputs = _lenient
except Exception:
    pass
