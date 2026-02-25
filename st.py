try:
    import minian
    assert minian.__version__ != None
except:
    print(1)
else:
    print(0)