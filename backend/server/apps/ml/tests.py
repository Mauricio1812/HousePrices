from django.test import TestCase

from apps.ml.regressor.cost_regressor import CostRegressor

import inspect
from apps.ml.registry import MLRegistry

class MLTests(TestCase):

    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)
        endpoint_name = "regressor"
        algorithm_object = CostRegressor()
        algorithm_name = "cost regressor"
        algorithm_status = "production"
        algorithm_version = "0.0.3"
        algorithm_owner = "Mauricio"
        algorithm_description = "Cost Regressor with simple pre- and post-processing"
        algorithm_code = inspect.getsource(CostRegressor)
        # add to registry
        registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
                    algorithm_status, algorithm_version, algorithm_owner,
                    algorithm_description, algorithm_code)
        # there should be one endpoint available
        self.assertEqual(len(registry.endpoints), 1)


    def test_rf_algorithm(self):
        input_data = { 
            "MSSubClass": 60 ,
            "MSZoning": "RL" ,
            "LotFrontage": 65 ,
            "LotArea": 8450 ,
            "Alley": "None" ,
            "LotShape": "Reg" ,
            "LandContour": "Lvl" ,
            "LotConfig": "Inside" ,
            "LandSlope": "Gtl" ,
            "Neighborhood": "CollgCr" ,
            "Condition1": "Norm" ,
            "Condition2": "Norm" ,
            "BldgType": "1Fam" ,
            "HouseStyle": "2Story",
            "OverallQual": 7 ,
            "OverallCond": 5 ,
            "YearBuilt": 2003 ,
            "YearRemodAdd": 2003 ,
            "RoofStyle": "Gable" ,
            "RoofMatl": "CompShg" ,
            "Exterior1st": "VinylSd" ,
            "Exterior2nd": "VinylSd" ,
            "MasVnrType": "BrkFace" ,
            "MasVnrArea": 196 ,
            "ExterQual": "Gd" ,
            "ExterCond": "TA" ,
            "Foundation": "PConc" ,
            "BsmtQual": "Gd" ,
            "BsmtCond": "TA" ,
            "BsmtExposure": "No" ,
            "BsmtFinType1": "GLQ" ,
            "BsmtFinSF1": 706 ,
            "BsmtFinType2": "Unf" ,
            "BsmtFinSF2": 0 ,
            "BsmtUnfSF": 150 ,
            "TotalBsmtSF": 856 ,
            "Heating": "GasA" ,
            "HeatingQC": "Ex" ,
            "CentralAir": "Y" ,
            "Electrical": "SBrkr" ,
            "1stFlrSF": 856 ,
            "2ndFlrSF": 854 ,
            "LowQualFinSF": 0 ,
            "GrLivArea": 1710 ,
            "BsmtFullBath": 1 ,
            "BsmtHalfBath": 0 ,
            "FullBath": 2 ,
            "HalfBath": 1 ,
            "BedroomAbvGr": 3 ,
            "KitchenAbvGr": 1 ,
            "KitchenQual": "Gd" ,
            "TotRmsAbvGrd": 8 ,
            "Functional": "Typ" ,
            "Fireplaces": 0 ,
            "FireplaceQu": "None" ,
            "GarageType": "Attchd" ,
            "GarageYrBlt": 2003 ,
            "GarageFinish": "RFn" ,
            "GarageCars": 2 ,
            "GarageArea": 548 ,
            "GarageQual": "TA" ,
            "GarageCond": "TA" ,
            "PavedDrive": "Y" ,
            "WoodDeckSF": 0 ,
            "OpenPorchSF": 61 ,
            "EnclosedPorch": 0 ,
            "3SsnPorch": 0 ,
            "ScreenPorch": 0 ,
            "PoolArea": 0 ,
            "Fence": "None" ,
            "MiscFeature": "None" ,
            "MiscVal": 0 ,
            "MoSold": 2 ,
            "YrSold": 2008 ,
            "SaleType": "WD" ,
            "SaleCondition": "Normal"
        }
        my_alg = CostRegressor()
        response = my_alg.compute_prediction(input_data)
        print(response)