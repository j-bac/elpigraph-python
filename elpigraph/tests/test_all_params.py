import pytest
import numpy as np
import matplotlib.pyplot as plt
import elpigraph

@pytest.fixture
def data():
    X = np.random.random((100,10))
    return X

#test default and non-default parameters 
def test_elpi_params(data):
    epg = elpigraph.computeElasticPrincipalTree(data,10)   
    