from proto.marshal.collections.maps import MapComposite
from proto.marshal.collections.repeated import RepeatedComposite

def to_native(obj):
    """Recursively convert protobuf-like objects (MapComposite, RepeatedComposite) to native Python types."""
    if isinstance(obj, MapComposite):
        return {k: to_native(v) for k, v in obj.items()}
    elif isinstance(obj, RepeatedComposite):
        return [to_native(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_native(v) for v in obj]
    else:
        return obj

