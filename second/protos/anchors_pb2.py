# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: second/protos/anchors.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='second/protos/anchors.proto',
  package='second.protos',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x1bsecond/protos/anchors.proto\x12\rsecond.protos\"r\n\x15\x41nchorGeneratorStride\x12\r\n\x05sizes\x18\x01 \x03(\x02\x12\x0f\n\x07strides\x18\x02 \x03(\x02\x12\x0f\n\x07offsets\x18\x03 \x03(\x02\x12\x11\n\trotations\x18\x04 \x03(\x02\x12\x15\n\rcustom_values\x18\x05 \x03(\x02\"f\n\x14\x41nchorGeneratorRange\x12\r\n\x05sizes\x18\x01 \x03(\x02\x12\x15\n\ranchor_ranges\x18\x02 \x03(\x02\x12\x11\n\trotations\x18\x03 \x03(\x02\x12\x15\n\rcustom_values\x18\x04 \x03(\x02\"\n\n\x08NoAnchorb\x06proto3')
)




_ANCHORGENERATORSTRIDE = _descriptor.Descriptor(
  name='AnchorGeneratorStride',
  full_name='second.protos.AnchorGeneratorStride',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='sizes', full_name='second.protos.AnchorGeneratorStride.sizes', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='strides', full_name='second.protos.AnchorGeneratorStride.strides', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='offsets', full_name='second.protos.AnchorGeneratorStride.offsets', index=2,
      number=3, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rotations', full_name='second.protos.AnchorGeneratorStride.rotations', index=3,
      number=4, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='custom_values', full_name='second.protos.AnchorGeneratorStride.custom_values', index=4,
      number=5, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=46,
  serialized_end=160,
)


_ANCHORGENERATORRANGE = _descriptor.Descriptor(
  name='AnchorGeneratorRange',
  full_name='second.protos.AnchorGeneratorRange',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='sizes', full_name='second.protos.AnchorGeneratorRange.sizes', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='anchor_ranges', full_name='second.protos.AnchorGeneratorRange.anchor_ranges', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rotations', full_name='second.protos.AnchorGeneratorRange.rotations', index=2,
      number=3, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='custom_values', full_name='second.protos.AnchorGeneratorRange.custom_values', index=3,
      number=4, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=162,
  serialized_end=264,
)


_NOANCHOR = _descriptor.Descriptor(
  name='NoAnchor',
  full_name='second.protos.NoAnchor',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=266,
  serialized_end=276,
)

DESCRIPTOR.message_types_by_name['AnchorGeneratorStride'] = _ANCHORGENERATORSTRIDE
DESCRIPTOR.message_types_by_name['AnchorGeneratorRange'] = _ANCHORGENERATORRANGE
DESCRIPTOR.message_types_by_name['NoAnchor'] = _NOANCHOR
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

AnchorGeneratorStride = _reflection.GeneratedProtocolMessageType('AnchorGeneratorStride', (_message.Message,), dict(
  DESCRIPTOR = _ANCHORGENERATORSTRIDE,
  __module__ = 'second.protos.anchors_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.AnchorGeneratorStride)
  ))
_sym_db.RegisterMessage(AnchorGeneratorStride)

AnchorGeneratorRange = _reflection.GeneratedProtocolMessageType('AnchorGeneratorRange', (_message.Message,), dict(
  DESCRIPTOR = _ANCHORGENERATORRANGE,
  __module__ = 'second.protos.anchors_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.AnchorGeneratorRange)
  ))
_sym_db.RegisterMessage(AnchorGeneratorRange)

NoAnchor = _reflection.GeneratedProtocolMessageType('NoAnchor', (_message.Message,), dict(
  DESCRIPTOR = _NOANCHOR,
  __module__ = 'second.protos.anchors_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.NoAnchor)
  ))
_sym_db.RegisterMessage(NoAnchor)


# @@protoc_insertion_point(module_scope)
