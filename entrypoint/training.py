from src.Pipelines.trainig_pipeline import TrainingPipeline

if __name__ == '__main__':
    train_model = TrainingPipeline()
    train_model.run_training_pipeline()