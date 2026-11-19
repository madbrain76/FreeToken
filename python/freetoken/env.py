from __future__ import annotations

import os
from functools import partial
from typing import Callable, Generic, TypeVar


class BaseEnv:
    def _init(self, name: str) -> None:
        raise NotImplementedError


T = TypeVar("T")


class EnvVar(BaseEnv, Generic[T]):
    def __init__(self, default_value: T, fn: Callable[[str], T]):
        self.value = default_value
        # What the build shipped, as opposed to what the environment made it: a guard's default
        # is a policy, and a test or a provenance line has to be able to read the policy without
        # re-deriving it from the (possibly patched) current value.
        self.default = default_value
        self.fn = fn
        super().__init__()

    def _init(self, name: str) -> None:
        env_value = os.getenv(name)
        if env_value is not None:
            try:
                self.value = self.fn(env_value)
            except Exception:
                pass

    def __bool__(self):
        return self.value

    def __str__(self):
        return str(self.value)


_TO_BOOL = lambda x: x.lower() in ("1", "true", "yes")


def _PARSE_MEM_BYTES(mem: str) -> int:
    mem = mem.strip().upper()
    if not mem[-1].isalpha():
        return int(mem)
    if mem.endswith("B"):
        mem = mem[:-1]
    UNIT_MAP = {"K": 1024, "M": 1024**2, "G": 1024**3}
    return int(float(mem[:-1]) * UNIT_MAP[mem[-1]])


ENV_PREFIX = "FREETOKEN_"
EnvInt = partial(EnvVar[int], fn=int)
EnvFloat = partial(EnvVar[float], fn=float)
EnvBool = partial(EnvVar[bool], fn=_TO_BOOL)
EnvOption = partial(EnvVar[bool | None], fn=_TO_BOOL, default_value=None)
EnvMem = partial(EnvVar[int], fn=_PARSE_MEM_BYTES)
EnvStr = partial(EnvVar[str], fn=str)


class EnvClassSingleton:
    _instance: EnvClassSingleton | None = None

    # shell
    SHELL_MAX_TOKENS = EnvInt(2048)
    # None = unset -> resolved from the model's generation_config.json sampling defaults
    # (sglang's sampling_defaults='model'); set the env var to override.
    SHELL_TOP_K = EnvInt(None)
    SHELL_TOP_P = EnvFloat(None)
    SHELL_TEMPERATURE = EnvFloat(None)

    # backend runtime
    FLASHINFER_USE_TENSOR_CORES = EnvOption()
    DISABLE_OVERLAP_SCHEDULING = EnvBool(False)
    PYNCCL_MAX_BUFFER_SIZE = EnvMem(1024**3)
    STEP_TIMEOUT = EnvFloat(120.0)
    # Idle seconds after which an expanded elastic KV pool shrinks back to --kv-initial-tokens
    # and restores the initial MoE slot cache; set to 0 to disable automatic idle shrink.
    ELASTIC_KV_SHRINK_IDLE_SEC = EnvFloat(120.0)
    DISTRIBUTED_TIMEOUT = EnvFloat(2592000.0)
    # Fail a request whose engine has gone silent, so a wedged backend cannot strand the
    # client forever: the surviving TP ranks sit in a collective allowed to run for
    # DISTRIBUTED_TIMEOUT (30 days) and no ack ever arrives. Measured between acks, not
    # end to end, so a queued request is only at risk if the whole engine stalls. 0
    # disables the breaker and restores the wait-forever behaviour.
    ACK_IDLE_TIMEOUT = EnvFloat(900.0)
    # Degenerate-generation breakers (freetoken/storm.py): fail a request whose logits
    # went non-finite, or that repeats one token this many times in a row, instead of
    # streaming token 0 ("!") for its whole output budget. 0 disables the run check.
    ABORT_ON_NONFINITE_LOGITS = EnvBool(True)
    STORM_TOKEN_RUN = EnvInt(1024)
    # Append-only JSONL record of every fault the breakers raise, so the evidence survives
    # a lost terminal. Empty disables the file and leaves logging as the only channel.
    STORM_EVENT_LOG = EnvStr("~/.freetoken/storm-events.jsonl")
    # Exit the server once this many DISTINCT requests have tripped a breaker inside
    # STORM_FATAL_WINDOW_S. One trip is a request the sampler could not serve; trips that keep
    # arriving on new requests after the quarantine has run mean the poison sits in state the
    # request does not own, and no later request can be trusted. 0 keeps serving and only logs.
    STORM_FATAL_TRIPS = EnvInt(8)
    STORM_FATAL_WINDOW_S = EnvFloat(300.0)
    # Exit the server once this many DISTINCT requests have tripped the *non-finite logits*
    # breaker inside STORM_FATAL_WINDOW_S, counted separately from STORM_FATAL_TRIPS. A wholly
    # non-finite logits row is not an unlucky sample: the value is already bad where the sampler
    # reads it, so the poison sits in state no request owns and every later request that touches
    # the same pages fails at the same offset. Observed 2026-11-10 -- two trips five seconds
    # apart and then nine silent hours, because the client had stopped retrying and a rule that
    # counts trips over a window could never fire. 1 takes the instance down on the first one;
    # 0 leaves non-finite rows to STORM_FATAL_TRIPS alone.
    STORM_FATAL_NONFINITE_TRIPS = EnvInt(1)
    # Give the prefix cache a private clone of a donated GDN snapshot instead of the request's
    # own slot. Transfer left one slot id owned by both the tree and a still-in-flight pipeline,
    # so a late write through a retained alias poisoned every later hit of that branch.
    LINEAR_STATE_COPY_ON_DONATE = EnvBool(True)
    # Clear a freshly allocated live GDN slot before the first forward reads it as carry-in.
    # A recycled slot holds whatever the previous owner left, including NaN.
    LINEAR_STATE_CLEAR_ON_ADMIT = EnvBool(True)
    # Diagnostic: hook every layer module and name the first one whose output went
    # non-finite, reported with the abort above. One reduction per layer per step.
    NAN_TRACE = EnvBool(False)
    # GatedDeltaNet recurrent (SSM) state dtype: float32 (default) | bfloat16 | float16.
    # fp32 matches the Qwen3.x configs (mamba_ssm_dtype); fp16/bf16 halves the GDN state
    # pool at some precision cost on the long recurrence (mirrors SGLang's mamba_ssm_dtype).
    MAMBA_SSM_DTYPE = EnvStr("float32")

    def __new__(cls):
        # single instance
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            attr_value = getattr(self, attr_name)
            assert isinstance(attr_value, BaseEnv)
            attr_value._init(f"{ENV_PREFIX}{attr_name}")


ENV = EnvClassSingleton()
