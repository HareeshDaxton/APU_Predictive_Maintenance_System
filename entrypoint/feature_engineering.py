from src.Pipelines.feature_engineering_pipeline import FeatureEngineering

if __name__ == "__main__":
    FE_pipeline = FeatureEngineering()
    FE_pipeline.run_feature_engineering()