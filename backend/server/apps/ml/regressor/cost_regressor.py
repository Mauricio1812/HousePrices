import pickle
import pandas as pd
import numpy as np
from scipy.special import boxcox1p

class CostRegressor:
    def __init__(self):
        path_to_ML = "../../pre_trained/"
        self.model = pickle.load(open(path_to_ML + 'HousePrice_model.sav','rb'))
        self.values_fill_missing =  pickle.load(open(path_to_ML + "train_mode.p", "rb"))
        #self.encoders = joblib.load(path_to_artifacts + "encoders.joblib")

    def preprocessing(self, input_data):
        numeric_feats = [
        {  'column ':  'FireplaceQu ', 'Ex': 0.0, 'Fa': 0.7304631471189666, 'Gd': 1.1943176378757767, 'None': 1.5409627556327752, 'Po': 1.8203341036428238, 'TA': 2.055641538058108},
        {  'column ':  'BsmtQual ', 'Ex': 0.0, 'Fa': 0.7304631471189666, 'Gd': 1.1943176378757767, 'None': 1.5409627556327752, 'TA': 1.8203341036428238},
        {  'column ':  'BsmtCond ', 'Fa': 0.0, 'Gd': 0.7304631471189666, 'None': 1.1943176378757767, 'Po': 1.5409627556327752, 'TA': 1.8203341036428238},
        {  'column ':  'GarageQual ', 'Ex': 0.0, 'Fa': 0.7304631471189666, 'Gd': 1.1943176378757767, 'None': 1.5409627556327752, 'Po': 1.8203341036428238, 'TA': 2.055641538058108},
        {  'column ':  'GarageCond ', 'Ex': 0.0, 'Fa': 0.7304631471189666, 'Gd': 1.1943176378757767, 'None': 1.5409627556327752, 'Po': 1.8203341036428238, 'TA': 2.055641538058108},
        {  'column ':  'ExterQual ', 'Ex': 0.0, 'Fa': 0.7304631471189666, 'Gd': 1.1943176378757767, 'TA': 1.5409627556327752},
        {  'column ':  'ExterCond ', 'Ex': 0.0, 'Fa': 0.7304631471189666, 'Gd': 1.1943176378757767, 'Po': 1.5409627556327752, 'TA': 1.8203341036428238},
        {  'column ':  'HeatingQC ', 'Ex': 0.0, 'Fa': 0.7304631471189666, 'Gd': 1.1943176378757767, 'Po': 1.5409627556327752, 'TA': 1.8203341036428238},
        {  'column ':  'KitchenQual ', 'Ex': 0.0, 'Fa': 0.7304631471189666, 'Gd': 1.1943176378757767, 'TA': 1.5409627556327752, 'nan': 1.8203341036428238},
        {  'column ':  'BsmtFinType1 ', 'ALQ': 0.0, 'BLQ': 0.7304631471189666, 'GLQ': 1.1943176378757767, 'LwQ': 1.5409627556327752, 'None': 1.8203341036428238, 'Rec': 2.055641538058108, 'Unf': 2.2596737867230705},
        {  'column ':  'BsmtFinType2 ', 'ALQ': 0.0, 'BLQ': 0.7304631471189666, 'GLQ': 1.1943176378757767, 'LwQ': 1.5409627556327752, 'None': 1.8203341036428238, 'Rec': 2.055641538058108, 'Unf': 2.2596737867230705},
        {  'column ':  'Functional ', 'Maj1': 0.0, 'Maj2': 0.7304631471189666, 'Min1': 1.1943176378757767, 'Min2': 1.5409627556327752, 'Mod': 1.8203341036428238, 'Sev': 2.055641538058108, 'Typ': 2.2596737867230705, 'nan': 2.440268378362637},
        {  'column ':  'Fence ', 'GdPrv': 0.0, 'GdWo': 0.7304631471189666, 'MnPrv': 1.1943176378757767, 'MnWw': 1.5409627556327752, 'None': 1.8203341036428238},
        {  'column ':  'BsmtExposure ', 'Av': 0.0, 'Gd': 0.7304631471189666, 'Mn': 1.1943176378757767, 'No': 1.5409627556327752, 'None': 1.8203341036428238},
        {  'column ':  'GarageFinish ', 'Fin': 0.0, 'None': 0.7304631471189666, 'RFn': 1.1943176378757767, 'Unf': 1.5409627556327752},
        {  'column ':  'LandSlope ', 'Gtl': 0.0, 'Mod': 0.7304631471189666, 'Sev': 1.1943176378757767},
        {  'column ':  'LotShape ', 'IR1': 0.0, 'IR2': 0.7304631471189666, 'IR3': 1.1943176378757767, 'Reg': 1.5409627556327752},
        {  'column ':  'PavedDrive ', 'N': 0.0, 'P': 0.7304631471189666, 'Y': 1.1943176378757767},
        {  'column ':  'Alley ', 'Grvl': 0.0, 'None': 0.7304631471189666, 'Pave': 1.1943176378757767},
        {  'column ':  'CentralAir ', 'N': 0.0, 'Y': 0.7304631471189666},
        {  'column ':  'MSSubClass ', '120': 0.0, '150': 0.7304631471189666, '160': 1.1943176378757767, '180': 1.5409627556327752, '190': 1.8203341036428238, '20': 2.055641538058108, '30': 2.2596737867230705, '40': 2.440268378362637, '45': 2.6025944687727294, '50': 2.750250297485029, '60': 2.885846472488202, '70': 3.011340243262834, '75': 3.128238685769432, '80': 3.2377281976988326, '85': 3.340760310539712, '90': 3.4381104434026533},
        {  'column ':  'OverallCond ', '1': 0.0, '2': 0.7304631471189666, '3': 1.1943176378757767, '4': 1.5409627556327752, '5': 1.8203341036428238, '6': 2.055641538058108, '7': 2.2596737867230705, '8': 2.440268378362637, '9': 2.6025944687727294},
        {  'column ':  'YrSold ', '2006': 0.0, '2007': 0.7304631471189666, '2008': 1.1943176378757767, '2009': 1.5409627556327752, '2010': 1.8203341036428238},
        {  'column ':  'MoSold ', '1': 0.0, '10': 0.7304631471189666, '11': 1.1943176378757767, '12': 1.5409627556327752, '2': 1.8203341036428238, '3': 2.055641538058108, '4': 2.2596737867230705, '5': 2.440268378362637, '6': 2.6025944687727294, '7': 2.750250297485029, '8': 2.885846472488202, '9': 3.011340243262834},
        ]

        dummy_list = ['MSZoning_C (all)', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL',
        'MSZoning_RM', 'LandContour_Bnk', 'LandContour_HLS',
        'LandContour_Low', 'LandContour_Lvl', 'LotConfig_Corner',
        'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_FR3',
        'LotConfig_Inside', 'Neighborhood_Blmngtn', 'Neighborhood_Blueste',
        'Neighborhood_BrDale', 'Neighborhood_BrkSide',
        'Neighborhood_ClearCr', 'Neighborhood_CollgCr',
        'Neighborhood_Crawfor', 'Neighborhood_Edwards',
        'Neighborhood_Gilbert', 'Neighborhood_IDOTRR',
        'Neighborhood_MeadowV', 'Neighborhood_Mitchel',
        'Neighborhood_NAmes', 'Neighborhood_NPkVill',
        'Neighborhood_NWAmes', 'Neighborhood_NoRidge',
        'Neighborhood_NridgHt', 'Neighborhood_OldTown',
        'Neighborhood_SWISU', 'Neighborhood_Sawyer',
        'Neighborhood_SawyerW', 'Neighborhood_Somerst',
        'Neighborhood_StoneBr', 'Neighborhood_Timber',
        'Neighborhood_Veenker', 'Condition1_Artery', 'Condition1_Feedr',
        'Condition1_Norm', 'Condition1_PosA', 'Condition1_PosN',
        'Condition1_RRAe', 'Condition1_RRAn', 'Condition1_RRNe',
        'Condition1_RRNn', 'Condition2_Artery', 'Condition2_Feedr',
        'Condition2_Norm', 'Condition2_PosA', 'Condition2_PosN',
        'Condition2_RRAe', 'Condition2_RRAn', 'Condition2_RRNn',
        'BldgType_1Fam', 'BldgType_2fmCon', 'BldgType_Duplex',
        'BldgType_Twnhs', 'BldgType_TwnhsE', 'HouseStyle_1.5Fin',
        'HouseStyle_1.5Unf', 'HouseStyle_1Story', 'HouseStyle_2.5Fin',
        'HouseStyle_2.5Unf', 'HouseStyle_2Story', 'HouseStyle_SFoyer',
        'HouseStyle_SLvl', 'RoofStyle_Flat', 'RoofStyle_Gable',
        'RoofStyle_Gambrel', 'RoofStyle_Hip', 'RoofStyle_Mansard',
        'RoofStyle_Shed', 'RoofMatl_CompShg', 'RoofMatl_Membran',
        'RoofMatl_Metal', 'RoofMatl_Roll', 'RoofMatl_Tar&Grv',
        'RoofMatl_WdShake', 'RoofMatl_WdShngl', 'Exterior1st_AsbShng',
        'Exterior1st_AsphShn', 'Exterior1st_BrkComm',
        'Exterior1st_BrkFace', 'Exterior1st_CBlock', 'Exterior1st_CemntBd',
        'Exterior1st_HdBoard', 'Exterior1st_ImStucc',
        'Exterior1st_MetalSd', 'Exterior1st_Plywood', 'Exterior1st_Stone',
        'Exterior1st_Stucco', 'Exterior1st_VinylSd', 'Exterior1st_Wd Sdng',
        'Exterior1st_WdShing', 'Exterior2nd_AsbShng',
        'Exterior2nd_AsphShn', 'Exterior2nd_Brk Cmn',
        'Exterior2nd_BrkFace', 'Exterior2nd_CBlock', 'Exterior2nd_CmentBd',
        'Exterior2nd_HdBoard', 'Exterior2nd_ImStucc',
        'Exterior2nd_MetalSd', 'Exterior2nd_Other', 'Exterior2nd_Plywood',
        'Exterior2nd_Stone', 'Exterior2nd_Stucco', 'Exterior2nd_VinylSd',
        'Exterior2nd_Wd Sdng', 'Exterior2nd_Wd Shng', 'MasVnrType_BrkCmn',
        'MasVnrType_BrkFace', 'MasVnrType_None', 'MasVnrType_Stone',
        'Foundation_BrkTil', 'Foundation_CBlock', 'Foundation_PConc',
        'Foundation_Slab', 'Foundation_Stone', 'Foundation_Wood',
        'Heating_Floor', 'Heating_GasA', 'Heating_GasW', 'Heating_Grav',
        'Heating_OthW', 'Heating_Wall', 'Electrical_FuseA',
        'Electrical_FuseF', 'Electrical_FuseP', 'Electrical_Mix',
        'Electrical_SBrkr', 'GarageType_2Types', 'GarageType_Attchd',
        'GarageType_Basment', 'GarageType_BuiltIn', 'GarageType_CarPort',
        'GarageType_Detchd', 'GarageType_None', 'MiscFeature_Gar2',
        'MiscFeature_None', 'MiscFeature_Othr', 'MiscFeature_Shed',
        'MiscFeature_TenC', 'SaleType_COD', 'SaleType_CWD', 'SaleType_Con',
        'SaleType_ConLD', 'SaleType_ConLI', 'SaleType_ConLw',
        'SaleType_New', 'SaleType_Oth', 'SaleType_WD',
        'SaleCondition_Abnorml', 'SaleCondition_AdjLand',
        'SaleCondition_Alloca', 'SaleCondition_Family',
        'SaleCondition_Normal', 'SaleCondition_Partial']

        object_list = ['MSZoning', 'LandContour', 'LotConfig', 'Neighborhood',
        'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle',
        'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
        'Foundation', 'Heating', 'Electrical', 'GarageType', 'MiscFeature',
        'SaleType', 'SaleCondition']

        # JSON to pandas DataFrame
        input_data = pd.DataFrame(input_data, index=[0])
        # fill missing values
        input_data.fillna(self.values_fill_missing)

        for c in ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
            'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1', 
            'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
            'LotShape', 'PavedDrive', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
            'YrSold', 'MoSold']:
                feat = next((item for item in numeric_feats if c in item["column "]), False)
                key=str(input_data.iloc[0, input_data.columns.get_loc(c)])
                input_data.iloc[0, input_data.columns.get_loc(c)] = feat[key]

        input_data['TotalSF'] = input_data['TotalBsmtSF'] + input_data['1stFlrSF'] + input_data['2ndFlrSF']
        for c in ['LotFrontage', 'LotArea', 'OverallQual', 'YearBuilt', 'YearRemodAdd',
        'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
        '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
        'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
        'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea',
        'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
        'ScreenPorch', 'PoolArea', 'MiscVal', 'TotalSF']:
            input_data[c]=boxcox1p(input_data[c], 0.15)

        x_temp = pd.DataFrame(np.zeros((1, len(dummy_list))), columns = dummy_list)  

        for object_feat in object_list:
            for dummy_feat in dummy_list:
                if(object_feat in dummy_feat and input_data.iloc[0, input_data.columns.get_loc(object_feat)] in dummy_feat):
                    x_temp[dummy_feat]=1

        input_data = input_data.drop(columns=object_list)
        input_data = pd.concat([input_data, x_temp], axis=1)

        return input_data

    def predict(self, input_data):
        return np.expm1(self.model.predict(input_data.values))

    # def postprocessing(self, input_data):
    #     label = "<=50K"
    #     if input_data[1] > 0.5:
    #         label = ">50K"
    #     return {"probability": input_data[1], "label": label, "status": "OK"}

    def compute_prediction(self, input_data):
        try:
            input_data = self.preprocessing(input_data)
            prediction = self.predict(input_data)[0]  # only one sample
            #prediction = self.postprocessing(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return prediction