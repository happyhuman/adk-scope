"""Dataclasses mirroring the ADK Scope Feature proto."""

import dataclasses
import enum
import json
from typing import List


class Maturity(enum.Enum):
  """Represents the maturity of a feature."""

  STABLE = 0
  EXPERIMENTAL = 1
  BETA = 2
  DEPRECATED = 3
  UNKNOWN = 4


class Type(enum.Enum):
  """Represents the type of a feature."""

  FUNCTION = 0
  INSTANCE_METHOD = 1
  CLASS_METHOD = 2
  CONSTRUCTOR = 3


class ParamType(enum.Enum):
  """Represents the type of a parameter."""

  INT = 0
  FLOAT = 1
  STRING = 2
  LIST = 3
  SET = 4
  DICT = 5
  BOOL = 6
  ANY = 7


@dataclasses.dataclass
class Param:
  """Represents a parameter of a feature."""

  original_name: str
  normalized_name: str
  original_types: List[str] = dataclasses.field(default_factory=list)
  normalized_types: List[ParamType] = dataclasses.field(default_factory=list)
  description: str = ""
  is_optional: bool = False


@dataclasses.dataclass
class Feature:
  """Represents a single feature in ADK Scope."""

  # --- 1. IDENTITY & GROUPING ---
  original_name: str
  normalized_name: str
  description: str = ""
  member_of: str = "null"
  normalized_member_of: str = "null"

  # --- 2. CONTEXT ---
  maturity: Maturity = Maturity.UNKNOWN
  type: Type = Type.FUNCTION

  file_path: str = ""
  namespace: str = ""
  normalized_namespace: str = ""

  # --- 3. CONTRACT ---
  parameters: List[Param] = dataclasses.field(default_factory=list)

  original_return_types: List[str] = dataclasses.field(default_factory=list)
  normalized_return_types: List[ParamType] = dataclasses.field(
      default_factory=list
  )

  blocking: bool = True


@dataclasses.dataclass
class FeatureRegistry:
  """Represents a collection of features for a given language and version."""

  language: str
  version: str
  features: List[Feature] = dataclasses.field(default_factory=list)


class DataclassJSONEncoder(json.JSONEncoder):
  """JSON encoder for dataclasses and enums."""

  def default(self, o):
    if dataclasses.is_dataclass(o):
      return dataclasses.asdict(o)
    if isinstance(o, enum.Enum):
      return o.name
    return super().default(o)


def to_json(data, indent=None) -> str:
  """Converts a dataclass instance to a JSON string."""
  return json.dumps(data, cls=DataclassJSONEncoder, indent=indent)
