import keras
from keras import backend as K
import onnxmltools

def train_step(input_data, output_data, model_path, out_model_path, epoch_count, learning_rate = None):
    model = keras.models.load_model(model_path)
    if not learning_rate is None:
        K.set_value(model.optimizer.learning_rate, learning_rate)

    losses = model.train_on_batch(input_data, output_data)

    model.save(out_model_path)
    onnx_model = onnxmltools.convert_keras(model, target_opset=18)
    onnxmltools.utils.save_model(onnx_model, out_model_path + ".onnx")