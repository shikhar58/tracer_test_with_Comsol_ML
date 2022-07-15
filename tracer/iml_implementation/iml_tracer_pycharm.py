import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('C:\\Users\\shikhar\\PycharmProjects\\particle')
sys.path.append('D:\\ML\\IML_ai\\iml')
from iml.source.training_engines import NeuralNetworkTrainingEngine


if __name__ == '__main__':
    data_df = pd.read_csv('C:\\Users\\shikhar\\PycharmProjects\\particle\\final_constv_def.csv')
    input_df = data_df.iloc[:,:-2]
    output_df = data_df.iloc[:,-1]

    training_engine = NeuralNetworkTrainingEngine(input_dataset=input_df,
                                                        output_dataset=output_df,
                                                        )
    training_engine.run(filename='mymodel.h5',
                        batch_size=10,
                        epochs=150,
                        test_size=0.25,
                        clean_directory=False,
                        normalize=False,
                                )

    training_engine.export_summary()
    y_pred=training_engine.model.predict(training_engine.x_test)
    idx = 500
    aa = [x for x in range(idx)]
    plt.figure(figsize=(8, 4))
    plt.plot(aa, training_engine.y_test_original[:idx], marker='.', label="actual")
    plt.plot(aa, y_pred[:idx], 'r', label="prediction")