def validate_data_inputs(t,x,y,v):
    if x is None:
        assert y is not None
        assert v is not None
        assert len(t) == len(v)
        assert len(t) == len(y)
    if y is None:
        assert x is not None
        assert len(t) == len(x)
    if x is not None:
        assert y is None
        assert v is not None

def check_is_partial_data(t,x,y,v):
    validate_data_inputs(t,x,y,v)
    if v is None:
        return False
    else:
        return True