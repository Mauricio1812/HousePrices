# pip install scikit-learn==1.0.2

import pickle
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold
import numpy as np
#import warnings
#warnings.filterwarnings('ignore')

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)

def make_a_prediction(House):
    
    x = pd.DataFrame.from_dict([House])
        
    # read in the model
    my_model = pickle.load(open('../Tests/HousePrice_model.sav','rb'))
    # make a prediction
    prediction = my_model.predict(x.values)
    return(np.expm1(prediction))


if __name__ == "__main__":

    House = {
        "MSSubClass": 2.885846472488202 ,
        "LotFrontage": 5.831327892742091 ,
        "LotArea": 19.212182313188084 ,
        "Alley": 0.7304631471189666 ,
        "LotShape": 1.5409627556327752 ,
        "LandSlope": 0.0 ,
        "OverallQual": 2.440268378362637 ,
        "OverallCond": 1.8203341036428238 ,
        "YearBuilt": 14.187526814497952 ,
        "YearRemodAdd": 14.187526814497952 ,
        "MasVnrArea": 8.059125775508376 ,
        "ExterQual": 1.1943176378757767 ,
        "ExterCond": 1.8203341036428238 ,
        "BsmtQual": 1.1943176378757767 ,
        "BsmtCond": 1.8203341036428238 ,
        "BsmtExposure": 1.5409627556327752 ,
        "BsmtFinType1": 1.1943176378757767 ,
        "BsmtFinSF1": 11.170326790619496 ,
        "BsmtFinType2": 2.2596737867230705 ,
        "BsmtFinSF2": 0.0 ,
        "BsmtUnfSF": 7.483295642295591 ,
        "TotalBsmtSF": 11.69262258528247 ,
        "HeatingQC": 0.0 ,
        "CentralAir": 0.7304631471189666 ,
        "1stFlrSF": 11.69262258528247 ,
        "2ndFlrSF": 11.686189379390555 ,
        "LowQualFinSF": 0.0 ,
        "GrLivArea": 13.698887979000148 ,
        "BsmtFullBath": 0.7304631471189666 ,
        "BsmtHalfBath": 0.0 ,
        "FullBath": 1.1943176378757767 ,
        "HalfBath": 0.7304631471189666 ,
        "BedroomAbvGr": 1.5409627556327752 ,
        "KitchenAbvGr": 0.7304631471189666 ,
        "KitchenQual": 1.1943176378757767 ,
        "TotRmsAbvGrd": 2.6025944687727294 ,
        "Functional": 2.2596737867230705 ,
        "Fireplaces": 0.0 ,
        "FireplaceQu": 1.5409627556327752 ,
        "GarageYrBlt": 14.187526814497952 ,
        "GarageFinish": 1.1943176378757767 ,
        "GarageCars": 1.1943176378757767 ,
        "GarageArea": 10.506270963009685 ,
        "GarageQual": 2.055641538058108 ,
        "GarageCond": 2.055641538058108 ,
        "PavedDrive": 1.1943176378757767 ,
        "WoodDeckSF": 0.0 ,
        "OpenPorchSF": 5.7146689026506605 ,
        "EnclosedPorch": 0.0 ,
        "3SsnPorch": 0.0 ,
        "ScreenPorch": 0.0 ,
        "PoolArea": 0.0 ,
        "Fence": 1.8203341036428238 ,
        "MiscVal": 0.0 ,
        "MoSold": 1.8203341036428238 ,
        "YrSold": 1.1943176378757767 ,
        "TotalSF": 14.976590572395713 ,
        "MSZoning_C (all)": 0 ,
        "MSZoning_FV": 0 ,
        "MSZoning_RH": 0 ,
        "MSZoning_RL": 1 ,
        "MSZoning_RM": 0 ,
        "LandContour_Bnk": 0 ,
        "LandContour_HLS": 0 ,
        "LandContour_Low": 0 ,
        "LandContour_Lvl": 1 ,
        "LotConfig_Corner": 0 ,
        "LotConfig_CulDSac": 0 ,
        "LotConfig_FR2": 0 ,
        "LotConfig_FR3": 0 ,
        "LotConfig_Inside": 1 ,
        "Neighborhood_Blmngtn": 0 ,
        "Neighborhood_Blueste": 0 ,
        "Neighborhood_BrDale": 0 ,
        "Neighborhood_BrkSide": 0 ,
        "Neighborhood_ClearCr": 0 ,
        "Neighborhood_CollgCr": 1 ,
        "Neighborhood_Crawfor": 0 ,
        "Neighborhood_Edwards": 0 ,
        "Neighborhood_Gilbert": 0 ,
        "Neighborhood_IDOTRR": 0 ,
        "Neighborhood_MeadowV": 0 ,
        "Neighborhood_Mitchel": 0 ,
        "Neighborhood_NAmes": 0 ,
        "Neighborhood_NPkVill": 0 ,
        "Neighborhood_NWAmes": 0 ,
        "Neighborhood_NoRidge": 0 ,
        "Neighborhood_NridgHt": 0 ,
        "Neighborhood_OldTown": 0 ,
        "Neighborhood_SWISU": 0 ,
        "Neighborhood_Sawyer": 0 ,
        "Neighborhood_SawyerW": 0 ,
        "Neighborhood_Somerst": 0 ,
        "Neighborhood_StoneBr": 0 ,
        "Neighborhood_Timber": 0 ,
        "Neighborhood_Veenker": 0 ,
        "Condition1_Artery": 0 ,
        "Condition1_Feedr": 0 ,
        "Condition1_Norm": 1 ,
        "Condition1_PosA": 0 ,
        "Condition1_PosN": 0 ,
        "Condition1_RRAe": 0 ,
        "Condition1_RRAn": 0 ,
        "Condition1_RRNe": 0 ,
        "Condition1_RRNn": 0 ,
        "Condition2_Artery": 0 ,
        "Condition2_Feedr": 0 ,
        "Condition2_Norm": 1 ,
        "Condition2_PosA": 0 ,
        "Condition2_PosN": 0 ,
        "Condition2_RRAe": 0 ,
        "Condition2_RRAn": 0 ,
        "Condition2_RRNn": 0 ,
        "BldgType_1Fam": 1 ,
        "BldgType_2fmCon": 0 ,
        "BldgType_Duplex": 0 ,
        "BldgType_Twnhs": 0 ,
        "BldgType_TwnhsE": 0 ,
        "HouseStyle_1.5Fin": 0 ,
        "HouseStyle_1.5Unf": 0 ,
        "HouseStyle_1Story": 0 ,
        "HouseStyle_2.5Fin": 0 ,
        "HouseStyle_2.5Unf": 0 ,
        "HouseStyle_2Story": 1 ,
        "HouseStyle_SFoyer": 0 ,
        "HouseStyle_SLvl": 0 ,
        "RoofStyle_Flat": 0 ,
        "RoofStyle_Gable": 1 ,
        "RoofStyle_Gambrel": 0 ,
        "RoofStyle_Hip": 0 ,
        "RoofStyle_Mansard": 0 ,
        "RoofStyle_Shed": 0 ,
        "RoofMatl_CompShg": 1 ,
        "RoofMatl_Membran": 0 ,
        "RoofMatl_Metal": 0 ,
        "RoofMatl_Roll": 0 ,
        "RoofMatl_Tar&Grv": 0 ,
        "RoofMatl_WdShake": 0 ,
        "RoofMatl_WdShngl": 0 ,
        "Exterior1st_AsbShng": 0 ,
        "Exterior1st_AsphShn": 0 ,
        "Exterior1st_BrkComm": 0 ,
        "Exterior1st_BrkFace": 0 ,
        "Exterior1st_CBlock": 0 ,
        "Exterior1st_CemntBd": 0 ,
        "Exterior1st_HdBoard": 0 ,
        "Exterior1st_ImStucc": 0 ,
        "Exterior1st_MetalSd": 0 ,
        "Exterior1st_Plywood": 0 ,
        "Exterior1st_Stone": 0 ,
        "Exterior1st_Stucco": 0 ,
        "Exterior1st_VinylSd": 1 ,
        "Exterior1st_Wd Sdng": 0 ,
        "Exterior1st_WdShing": 0 ,
        "Exterior2nd_AsbShng": 0 ,
        "Exterior2nd_AsphShn": 0 ,
        "Exterior2nd_Brk Cmn": 0 ,
        "Exterior2nd_BrkFace": 0 ,
        "Exterior2nd_CBlock": 0 ,
        "Exterior2nd_CmentBd": 0 ,
        "Exterior2nd_HdBoard": 0 ,
        "Exterior2nd_ImStucc": 0 ,
        "Exterior2nd_MetalSd": 0 ,
        "Exterior2nd_Other": 0 ,
        "Exterior2nd_Plywood": 0 ,
        "Exterior2nd_Stone": 0 ,
        "Exterior2nd_Stucco": 0 ,
        "Exterior2nd_VinylSd": 1 ,
        "Exterior2nd_Wd Sdng": 0 ,
        "Exterior2nd_Wd Shng": 0 ,
        "MasVnrType_BrkCmn": 0 ,
        "MasVnrType_BrkFace": 1 ,
        "MasVnrType_None": 0 ,
        "MasVnrType_Stone": 0 ,
        "Foundation_BrkTil": 0 ,
        "Foundation_CBlock": 0 ,
        "Foundation_PConc": 1 ,
        "Foundation_Slab": 0 ,
        "Foundation_Stone": 0 ,
        "Foundation_Wood": 0 ,
        "Heating_Floor": 0 ,
        "Heating_GasA": 1 ,
        "Heating_GasW": 0 ,
        "Heating_Grav": 0 ,
        "Heating_OthW": 0 ,
        "Heating_Wall": 0 ,
        "Electrical_FuseA": 0 ,
        "Electrical_FuseF": 0 ,
        "Electrical_FuseP": 0 ,
        "Electrical_Mix": 0 ,
        "Electrical_SBrkr": 1 ,
        "GarageType_2Types": 0 ,
        "GarageType_Attchd": 1 ,
        "GarageType_Basment": 0 ,
        "GarageType_BuiltIn": 0 ,
        "GarageType_CarPort": 0 ,
        "GarageType_Detchd": 0 ,
        "GarageType_None": 0 ,
        "MiscFeature_Gar2": 0 ,
        "MiscFeature_None": 1 ,
        "MiscFeature_Othr": 0 ,
        "MiscFeature_Shed": 0 ,
        "MiscFeature_TenC": 0 ,
        "SaleType_COD": 0 ,
        "SaleType_CWD": 0 ,
        "SaleType_Con": 0 ,
        "SaleType_ConLD": 0 ,
        "SaleType_ConLI": 0 ,
        "SaleType_ConLw": 0 ,
        "SaleType_New": 0 ,
        "SaleType_Oth": 0 ,
        "SaleType_WD": 1 ,
        "SaleCondition_Abnorml": 0 ,
        "SaleCondition_AdjLand": 0 ,
        "SaleCondition_Alloca": 0 ,
        "SaleCondition_Family": 0 ,
        "SaleCondition_Normal": 1 ,
        "SaleCondition_Partial": 0 
    }

    cost = make_a_prediction(House)

    print(cost)