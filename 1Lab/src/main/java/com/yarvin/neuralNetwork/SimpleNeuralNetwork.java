package com.yarvin.neuralNetwork;

import com.yarvin.util.ModelUtils;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class SimpleNeuralNetwork {
    private static final Logger logger = LoggerFactory.getLogger(SimpleNeuralNetwork.class);

    public static void main(String[] args) throws Exception {
        int batchSize = 64;
        int numClasses = 10;
        int numEpochs = 100;

        DataSetIterator trainData = new MnistDataSetIterator(batchSize, true, 12345);
        DataSetIterator testData = new MnistDataSetIterator(batchSize, false, 12345);


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(784)
                        .nOut(1024)
                        .activation(Activation.RELU)
                        .dropOut(0.5)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(1024)
                        .nOut(512)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(512)
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

        for (int i = 0; i < numEpochs; i++) {
            logger.info("Эпоха " + (i + 1) + " из " + numEpochs);
            model.fit(trainData);
        }

        File modelFile = new File("model_28x28_2.zip");
        ModelSerializer.writeModel(model, modelFile, true);
        logger.info("Модель сохранена в файл: " + modelFile.getAbsolutePath());

        logger.info("Оценка модели на тестовых данных...");
        var eval = model.evaluate(testData);
        logger.info(eval.stats());
    }
}