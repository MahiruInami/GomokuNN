import keras
import tensorflow
from keras import backend as K
import onnxmltools

def train_step(input_data, output_data, model_path, out_model_path, epoch_count, batch_size, learning_rate = None):
    model = keras.models.load_model(model_path)
    if not learning_rate is None:
        K.set_value(model.optimizer.learning_rate, learning_rate)

    early_stoping_callback = tensorflow.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    history = model.fit(input_data, output_data, epochs=epoch_count, batch_size=batch_size, callbacks=[early_stoping_callback])

    model.save(out_model_path)
    onnx_model = onnxmltools.convert_keras(model, target_opset=18)
    onnxmltools.utils.save_model(onnx_model, out_model_path + ".onnx")