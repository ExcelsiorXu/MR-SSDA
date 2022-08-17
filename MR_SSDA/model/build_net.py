import sys
sys.path.append('./model')
import BreNet_MLFF

def Generator(model):
    if model == 'BreNet_MLFF':
        return BreNet_MLFF.Feature()


def Classifier(model):
    if model == 'BreNet_MLFF':
        return BreNet_MLFF.Predictor()

