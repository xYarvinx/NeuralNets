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
    //private static final Logger logger = LoggerFactory.getLogger(SimpleNeuralNetwork.class);

    public static void main(String[] args) throws Exception {
        int batchSize = 64;
        int numClasses = 10;
        int numEpochs = 500;

        // Загрузка данных MNIST
        DataSetIterator mnistTrainData = new MnistDataSetIterator(batchSize, true, 12345);
        DataSetIterator mnistTestData = new MnistDataSetIterator(batchSize, false, 12345);

        // Преобразование данных в 8x8
        List<org.nd4j.linalg.dataset.DataSet> trainDataList = resizeData(mnistTrainData, 8, 8);
        List<org.nd4j.linalg.dataset.DataSet> testDataList = resizeData(mnistTestData, 8, 8);

        DataSetIterator trainData = new ListDataSetIterator<>(trainDataList);
        DataSetIterator testData = new ListDataSetIterator<>(testDataList);

        // Конфигурация нейронной сети
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(64)
                        .nOut(512)
                        .activation(Activation.RELU)
                        .dropOut(0.5)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(512)
                        .nOut(256)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(256)
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        // Создание и инициализация модели
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

        // Обучение модели
        for (int i = 0; i < numEpochs; i++) {
            System.out.println("Эпоха " + (i + 1) + " из " + numEpochs);
            model.fit(trainData);
        }

        // Сохранение модели в файл
        File modelFile = new File("model_8x8.zip");
        ModelSerializer.writeModel(model, modelFile, true);
        System.out.println("Модель сохранена в файл: " + modelFile.getAbsolutePath());

        // Оценка модели на тестовых данных
        System.out.println("Оценка модели на тестовых данных...");
        var eval = model.evaluate(testData);
        System.out.println(eval.stats());
    }

    private static List<org.nd4j.linalg.dataset.DataSet> resizeData(DataSetIterator iterator, int newWidth, int newHeight) {
        List<org.nd4j.linalg.dataset.DataSet> resizedData = new ArrayList<>();
        while (iterator.hasNext()) {
            org.nd4j.linalg.dataset.DataSet dataSet = iterator.next();
            org.nd4j.linalg.dataset.DataSet resizedDataSet = ModelUtils.resizeDataSet(dataSet, newWidth, newHeight);
            resizedData.add(resizedDataSet);
        }
        return resizedData;
    }
}