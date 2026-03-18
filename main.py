from Kidney_Tumor_Classifier import logger
from Kidney_Tumor_Classifier.pipeline.stage_01 import DataIngestionTrainingPipeline
from Kidney_Tumor_Classifier.pipeline.stage_02 import PrepareBaseModelTrainingPipeline




STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started ") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed \n")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Prepare base model"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started ")
   prepare_base_model = PrepareBaseModelTrainingPipeline()
   prepare_base_model.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed \n")
except Exception as e:
        logger.exception(e)
        raise e