def clean_object(
    d: dict[str, object] | list[object]
) -> dict[str, object] | list[object]:
    if isinstance(d, dict):
        for k, v in list(d.items()):
            if not v:
                del d[k]
            elif isinstance(v, (dict, list)):
                clean_object(v)  # type: ignore
    else:
        for v in d:
            if not v:
                d.remove(v)
            elif isinstance(v, (dict, list)):
                clean_object(v)  # type: ignore
    return d
