import yaml
from pipeline.inference import Pipeline


def init_inference_pipeline():
    # get pipeline settings from config file
    pipeline_settings = yaml.safe_load(open("pipeline_inference_config.yml"))
    cogwheel_type = pipeline_settings['COGWHEEL_TYPE']
    model_type = pipeline_settings['MODEL_TYPE']
    batch_size = pipeline_settings['BATCH_SIZE']

    # grab checkpoints from checkpoint structure  #Pai change path
    ae_checkpoint_dir = f'./checkpoints_skeleton/autoencoder/training/{cogwheel_type}/checkpoints'
    cls_checkpoint_dir = f'./checkpoints_skeleton/second_stage/training/{cogwheel_type}/{model_type}/checkpoints'

    # init pipeline
    pipeline = Pipeline(model_type=model_type,
                        cogwheel_type=cogwheel_type,
                        ae_checkpoint_dir=ae_checkpoint_dir,
                        cls_checkpoint_dir=cls_checkpoint_dir,
                        batch_size=batch_size)

    # return initialized inference pipeline
    return pipeline


if __name__ == '__main__':
    import cv2

    pipeline = init_inference_pipeline()
    demo_image_path = './CAM1_20200615_105331_000066_066_5P8_NG.png'
    img = cv2.imread(demo_image_path, 0)
    bboxes = pipeline.inference(img)
    print(bboxes)