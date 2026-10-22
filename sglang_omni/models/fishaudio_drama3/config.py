# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Fish Audio Drama 3 TTS."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field, PositiveInt

from sglang_omni.config.placement import StagePlacementPlan, StagePlacementPlanner
from sglang_omni.config.schema import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)
from sglang_omni.config.topology import compile_logical_processes
from sglang_omni.pipeline.replicas import expand_replica_stages

PACKAGE = "sglang_omni.models.fishaudio_drama3"
DEFAULT_CONTEXT_LENGTH = 16384
DEFAULT_MAX_RUNNING_REQUESTS = 16
# A request that sets no max_new_tokens fills the remaining context up to this.
DEFAULT_MAX_NEW_TOKENS_CAP = 2048
# A prompt with more rows is refused before admission (HTTP 400).
DEFAULT_MAX_PROMPT_ROWS = 8192
DEFAULT_STREAM_STRIDE = 10
DEFAULT_STREAM_FOLLOWUP_STRIDE_RAMP = (10, 12, 32)
DEFAULT_STREAM_FOLLOWUP_STRIDE = 64
DEFAULT_FIRST_FRAME_CUDA_GRAPHS = True
DEFAULT_FIRST_FRAME_CUDA_GRAPH_MAX_REQUESTS = 16
DEFAULT_TRITON_SAMPLING = True
DEFAULT_TOWER_FUSED_LAYERS = True
DEFAULT_ENABLE_ASYNC_DECODE = False
DEFAULT_ASYNC_DECODE_MIN_BATCH_SIZE = 1
# The codec of the public S2-Pro release, pinned to an immutable commit.
DEFAULT_CODEC_PATH = "fishaudio/s2-pro@1de9996b6be38b745688de084d87a5633f714e4e"
PREPROCESSING_TOTAL_RESERVE_BYTES = "6GiB"
TTS_ENGINE_GPU_MEMORY_FRACTION = 0.70
VOCODER_GPU_MEMORY_FRACTION = 0.10
VOCODER_WINDOW_FIELDS = (
    "stream_stride",
    "stream_followup_stride_ramp",
    "stream_followup_stride",
)


class Drama3PreprocessingFactoryArgs(FactoryArgs):
    max_concurrency: int = Field(default=8, ge=1)
    reference_encode_dtype: Literal["float32", "bfloat16"] = "bfloat16"
    reference_encode_cuda_streams: Literal[0, 2] = 0
    max_reference_audio_seconds: float = Field(default=60.0, gt=0)


class Drama3PreprocessingStageConfig(StageConfig):
    factory: Drama3PreprocessingFactoryArgs = Field(
        default_factory=Drama3PreprocessingFactoryArgs
    )


class Drama3EngineFactoryArgs(FactoryArgs):
    """Engine knobs; max_new_tokens is the most a request may set and bounds one that
    sets none, and preprocessing refuses prompts by the same max_prompt_rows."""

    context_length: int = Field(default=DEFAULT_CONTEXT_LENGTH, gt=0)
    max_running_requests: int = Field(default=DEFAULT_MAX_RUNNING_REQUESTS, ge=1)
    max_new_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS_CAP, gt=0)
    max_prompt_rows: int = Field(default=DEFAULT_MAX_PROMPT_ROWS, ge=2)
    first_frame_cuda_graphs: bool = DEFAULT_FIRST_FRAME_CUDA_GRAPHS
    first_frame_cuda_graph_max_requests: int = Field(
        default=DEFAULT_FIRST_FRAME_CUDA_GRAPH_MAX_REQUESTS, ge=1
    )
    triton_sampling: bool = DEFAULT_TRITON_SAMPLING
    tower_fused_layers: bool = DEFAULT_TOWER_FUSED_LAYERS
    enable_async_decode: bool = DEFAULT_ENABLE_ASYNC_DECODE
    async_decode_min_batch_size: int = Field(
        default=DEFAULT_ASYNC_DECODE_MIN_BATCH_SIZE, ge=1
    )


class Drama3EngineStageConfig(EngineStageConfig):
    factory: Drama3EngineFactoryArgs = Field(default_factory=Drama3EngineFactoryArgs)


class Drama3VocoderFactoryArgs(FactoryArgs):
    """Codec path (shared with preprocessing) and the streaming window schedule, in frames."""

    codec_path: str = DEFAULT_CODEC_PATH
    stream_stride: int = Field(default=DEFAULT_STREAM_STRIDE, ge=1)
    stream_followup_stride_ramp: tuple[PositiveInt, ...] = (
        DEFAULT_STREAM_FOLLOWUP_STRIDE_RAMP
    )
    stream_followup_stride: int = Field(default=DEFAULT_STREAM_FOLLOWUP_STRIDE, ge=1)
    stream_overlap_tokens: int = Field(default=32, ge=0)
    stream_crossfade_samples: int = Field(default=0, ge=0)
    max_batch_size: int = Field(default=8, ge=1)
    max_batch_wait_ms: int = Field(default=2, ge=0)
    window_cuda_graphs: bool = True
    decode_dtype: Literal["float32", "float16"] = "float16"


class Drama3VocoderStageConfig(StageConfig):
    factory: Drama3VocoderFactoryArgs = Field(default_factory=Drama3VocoderFactoryArgs)


class Drama3PipelineConfig(PipelineConfig):
    """preprocessing → tts_engine → vocoder."""

    architecture: ClassVar[str] = "FishOmniModel"
    requires_model_capabilities: ClassVar[bool] = True
    supported_speech_fields: ClassVar[frozenset[str] | None] = frozenset(
        {
            "model",
            "input",
            "voice",
            "response_format",
            "speed",
            "stream",
            "stream_format",
            "references",
            "ref_audio",
            "ref_text",
            "task_type",
            "max_new_tokens",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "stream_audio",
            "split_granularity",
        }
    )
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "preprocessing": Drama3PreprocessingStageConfig,
        "tts_engine": Drama3EngineStageConfig,
        "vocoder": Drama3VocoderStageConfig,
    }

    placement_policy: str | None = (
        f"{PACKAGE}.config.validate_reference_encode_placement"
    )
    model_path: str
    stages: list[StageConfig] = [
        Drama3PreprocessingStageConfig(
            name="preprocessing",
            process="preprocessing",
            factory_path=f"{PACKAGE}.stages.create_preprocessing_executor",
            gpu=0,
            total_reserve_bytes=PREPROCESSING_TOTAL_RESERVE_BYTES,
            next="tts_engine",
        ),
        Drama3EngineStageConfig(
            name="tts_engine",
            process="pipeline",
            factory_path=f"{PACKAGE}.stages.create_tts_engine_executor",
            gpu=0,
            gpu_memory_fraction=TTS_ENGINE_GPU_MEMORY_FRACTION,
            next="vocoder",
            stream_to=["vocoder"],
        ),
        Drama3VocoderStageConfig(
            name="vocoder",
            process="vocoder",
            factory_path=f"{PACKAGE}.stages.create_vocoder_executor",
            gpu=0,
            gpu_memory_fraction=VOCODER_GPU_MEMORY_FRACTION,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]

    def model_post_init(self, __context: object = None) -> None:
        super().model_post_init(__context)
        preprocessing_extra = (
            self.stage_named("preprocessing").factory.model_extra or {}
        )
        for field_name, owner in (
            ("max_prompt_rows", "tts_engine"),
            ("codec_path", "vocoder"),
        ):
            if field_name in preprocessing_extra:
                raise ValueError(
                    f"Drama 3 preprocessing takes {field_name} from the {owner} "
                    f"stage; set stages.{owner}.factory.{field_name} instead"
                )
            else:
                pass
        engine_extra = self.stage_named("tts_engine").factory.model_extra or {}
        for field_name in VOCODER_WINDOW_FIELDS:
            if field_name in engine_extra:
                raise ValueError(
                    f"Drama 3's tts_engine takes {field_name} from the vocoder "
                    f"stage; set stages.vocoder.factory.{field_name} instead"
                )
            else:
                pass
        if self.stage_named("vocoder").factory.max_new_tokens is not None:
            raise ValueError(
                "Drama 3's vocoder takes max_new_tokens from the tts_engine stage; "
                "set stages.tts_engine.factory.max_new_tokens instead"
            )
        else:
            pass

        preprocessing = self.stage_named("preprocessing")
        assert isinstance(preprocessing.factory, Drama3PreprocessingFactoryArgs)
        if preprocessing.factory.reference_encode_cuda_streams == 2:
            if (
                self.placement_policy
                != f"{PACKAGE}.config.validate_reference_encode_placement"
            ):
                raise ValueError(
                    "two-stream reference encoding requires the Drama 3 placement_policy"
                )
            else:
                pass
            process_plan, stages = compile_logical_processes(self)
            expanded, replicas = expand_replica_stages(stages, process_plan)
            plan = StagePlacementPlanner(self).build(
                stages_cfg=expanded,
                apply_policy=False,
                replica_instances=replicas.replicas,
            )
            validate_reference_encode_placement(self, plan)
        else:
            pass

    def stage_factory_kwargs(
        self, stage_name: str
    ) -> dict[str, int | str | tuple[int, ...]]:
        engine_factory = self.stage_named("tts_engine").factory
        vocoder_factory = self.stage_named("vocoder").factory
        assert isinstance(engine_factory, Drama3EngineFactoryArgs)
        assert isinstance(vocoder_factory, Drama3VocoderFactoryArgs)
        if stage_name == "preprocessing":
            return {
                "max_prompt_rows": engine_factory.max_prompt_rows,
                "codec_path": vocoder_factory.codec_path,
            }
        elif stage_name == "tts_engine":
            return {
                "stream_stride": vocoder_factory.stream_stride,
                "stream_followup_stride_ramp": vocoder_factory.stream_followup_stride_ramp,
                "stream_followup_stride": vocoder_factory.stream_followup_stride,
            }
        elif stage_name == "vocoder":
            return {"max_new_tokens": engine_factory.max_new_tokens}
        else:
            return {}

    def supports_uploaded_voice_references(self) -> bool:
        return True


def validate_reference_encode_placement(
    config: PipelineConfig, plan: StagePlacementPlan
) -> None:
    preprocessing = config.stage_named("preprocessing")
    factory = preprocessing.factory
    assert isinstance(factory, Drama3PreprocessingFactoryArgs)
    if factory.reference_encode_cuda_streams == 0:
        return
    else:
        pass
    if factory.reference_encode_dtype != "bfloat16" or (
        factory.device is not None and factory.device.split(":")[0] != "cuda"
    ):
        raise ValueError("two-stream reference encoding requires CUDA and bfloat16")
    else:
        pass
    process_plan, _ = compile_logical_processes(config)
    process = process_plan.process_of("preprocessing")
    placements = plan.instances_of("preprocessing")
    if (
        process.stage_names != ("preprocessing",)
        or process.tp_size != 1
        or process.num_replicas != 1
        or len(placements) != 1
        or placements[0].tp_size != 1
        or len(placements[0].gpu_ids) != 1
    ):
        raise ValueError(
            "two-stream reference encoding requires a dedicated preprocessing "
            "process with TP1, one replica and one CUDA GPU"
        )
    else:
        pass
    preprocessing_gpu_ids = set(placements[0].gpu_ids)
    other_gpu_ids = {
        gpu_id
        for stage_name in ("tts_engine", "vocoder")
        for placement in plan.instances_of(stage_name)
        for gpu_id in placement.gpu_ids
    }
    if preprocessing_gpu_ids & other_gpu_ids:
        raise ValueError(
            "two-stream reference encoding requires an isolated preprocessing GPU "
            "not used by any tts_engine or vocoder instance"
        )
    else:
        pass


EntryClass = Drama3PipelineConfig
