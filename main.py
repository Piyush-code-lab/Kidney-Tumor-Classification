from Kidney_Tumor_Classifier import logger
from Kidney_Tumor_Classifier.pipeline.stage_01 import DataIngestionTrainingPipeline





STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started ") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed \n")
except Exception as e:
        logger.exception(e)
        raise e