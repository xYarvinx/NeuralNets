package com.yarvin;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.eval.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.learning.config.Adam;

import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.SwingWrapper;

import java.util.ArrayList;
import java.util.List;

public class FashionMnistExperiments {

    // Параметры модели
    private static final int seed = 123;
    private static final int batchSize = 64;
    private static final int numClasses = 10;
    private static final int inputSize = 28 * 28; 

    public static void main(String[] args) throws Exception {

        System.setProperty("MNIST_DATA_HOME", "resources/com/yarvin/fashionMnist");

        // Запуск всех экспериментов по очереди
        experimentEpochs();
        experimentLearningRate();
        experimentDropout();
        experimentBatchNormalization();
    }

    private static DataSetIterator getTrainIterator() throws Exception {
        DataSetIterator trainIter = new MnistDataSetIterator(batchSize, true, seed);
        DataNormalization scaler = new NormalizerMinMaxScaler(0, 1);
        scaler.fit(trainIter);
        trainIter.setPreProcessor(scaler);
        return trainIter;
    }

    private static DataSetIterator getTestIterator() throws Exception {
        DataSetIterator testIter = new MnistDataSetIterator(batchSize, false, seed);
        DataNormalization scaler = new NormalizerMinMaxScaler(0, 1);
        scaler.fit(testIter);
        testIter.setPreProcessor(scaler);
        return testIter;
    }

    /**
     * Эксперимент 1: Влияние числа эпох.
     * Обучаем модель с разным числом эпох (5 и 50).
     */
    private static void experimentEpochs() throws Exception {
        System.out.println("===== Эксперимент 1: Влияние числа эпох =====");
        int[] epochValues = {5, 50};
        for (int epochs : epochValues) {
            System.out.println("\nОбучение модели на " + epochs + " эпохах:");
            DataSetIterator trainIter = getTrainIterator();
            DataSetIterator testIter = getTestIterator();

            MultiLayerNetwork model = buildModel(false, false, 0.001);
            runTrainingAndPlot(model, trainIter, testIter, epochs, "Эпохи: " + epochs);
        }
    }

    /**
     * Эксперимент 2: Влияние learning rate.
     * Обучаем модель на 50 эпох с разными значениями learning rate.
     */
    private static void experimentLearningRate() throws Exception {
        System.out.println("===== Эксперимент 2: Влияние learning rate =====");
        double[] learningRates = {0.0005, 0.005};
        int epochs = 50;
        for (double lr : learningRates) {
            System.out.println("\nОбучение модели с learning rate = " + lr);
            DataSetIterator trainIter = getTrainIterator();
            DataSetIterator testIter = getTestIterator();
            MultiLayerNetwork model = buildModel(false, false, lr);
            runTrainingAndPlot(model, trainIter, testIter, epochs, "Learning Rate: " + lr);
        }
    }

    /**
     * Эксперимент 3: Влияние слоя dropout.
     * Сравниваем модель с dropout и без dropout на 50 эпох.
     */
    private static void experimentDropout() throws Exception {
        System.out.println("===== Эксперимент 3: Влияние слоя dropout =====");
        int epochs = 50;
        double lr = 0.001;

        System.out.println("\nОбучение модели с использованием dropout:");
        DataSetIterator trainIterDrop = getTrainIterator();
        DataSetIterator testIterDrop = getTestIterator();
        MultiLayerNetwork modelDrop = buildModel(true, false, lr);
        runTrainingAndPlot(modelDrop, trainIterDrop, testIterDrop, epochs, "С dropout");


        System.out.println("\nОбучение модели без использования dropout:");
        DataSetIterator trainIterNoDrop = getTrainIterator();
        DataSetIterator testIterNoDrop = getTestIterator();
        MultiLayerNetwork modelNoDrop = buildModel(false, false, lr);
        runTrainingAndPlot(modelNoDrop, trainIterNoDrop, testIterNoDrop, epochs, "Без dropout");
    }

    /**
     * Эксперимент 4: Влияние слоя batch normalization.
     * Сравниваем модель с batch normalization и без него на 50 эпох.
     */
    private static void experimentBatchNormalization() throws Exception {
        System.out.println("===== Эксперимент 4: Влияние слоя batch normalization =====");
        int epochs = 50;
        double lr = 0.001;

        System.out.println("\nОбучение модели с использованием batch normalization:");
        DataSetIterator trainIterBN = getTrainIterator();
        DataSetIterator testIterBN = getTestIterator();
        MultiLayerNetwork modelBN = buildModel(false, true, lr);
        runTrainingAndPlot(modelBN, trainIterBN, testIterBN, epochs, "С batch normalization");


        System.out.println("\nОбучение модели без использования batch normalization:");
        DataSetIterator trainIterNoBN = getTrainIterator();
        DataSetIterator testIterNoBN = getTestIterator();
        MultiLayerNetwork modelNoBN = buildModel(false, false, lr);
        runTrainingAndPlot(modelNoBN, trainIterNoBN, testIterNoBN, epochs, "Без batch normalization");
    }

    /**
     * Метод для построения модели.
     *
     * @param useDropout   использовать ли слой dropout
     * @param useBatchNorm использовать ли слой batch normalization
     * @param learningRate скорость обучения
     * @return сконфигурированная модель
     */
    private static MultiLayerNetwork buildModel(boolean useDropout, boolean useBatchNorm, double learningRate) {
        NeuralNetConfiguration.ListBuilder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(learningRate))
                .list();


        builder.layer(0, new DenseLayer.Builder()
                .nIn(inputSize)
                .nOut(256)
                .activation(Activation.RELU)
                .build());

        int layerIndex = 1;
        if (useDropout) {
            builder.layer(layerIndex++, new DropoutLayer.Builder(0.5).build());
        }
        if (useBatchNorm) {
            builder.layer(layerIndex++, new BatchNormalization.Builder().build());
        }
        builder.layer(layerIndex, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(numClasses)
                .activation(Activation.SOFTMAX)
                .build());

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));
        return model;
    }

    /**
     * Универсальный метод для обучения модели, сбора метрик и построения графиков.
     *
     * @param model          модель для обучения
     * @param trainIter      итератор обучающих данных
     * @param testIter       итератор тестовых данных
     * @param epochs         число эпох обучения
     * @param experimentName название эксперимента (для графиков)
     */
    private static void runTrainingAndPlot(MultiLayerNetwork model, DataSetIterator trainIter, DataSetIterator testIter,
                                           int epochs, String experimentName) throws Exception {
        List<Double> trainAccList = new ArrayList<>();
        List<Double> testAccList = new ArrayList<>();
        List<Double> trainLossList = new ArrayList<>();
        List<Integer> epochList = new ArrayList<>();

        for (int epoch = 1; epoch <= epochs; epoch++) {
            trainIter.reset();
            model.fit(trainIter);


            trainIter.reset();
            Evaluation evalTrain = model.evaluate(trainIter);
            double trainAcc = evalTrain.accuracy();


            testIter.reset();
            Evaluation evalTest = model.evaluate(testIter);
            double testAcc = evalTest.accuracy();

            double trainLoss = model.score();

            epochList.add(epoch);
            trainAccList.add(trainAcc);
            testAccList.add(testAcc);
            trainLossList.add(trainLoss);

            System.out.println("Эпоха " + epoch + " -> Точность на обучении: " + trainAcc
                    + ", Точность на тесте: " + testAcc + ", Значение ошибки: " + trainLoss);
        }

        plotChart(epochList, trainAccList, testAccList, "Точность - " + experimentName, "Эпоха", "Точность");
        plotChart(epochList, trainLossList, null, "Ошибка - " + experimentName, "Эпоха", "Ошибка");
    }

    /**
     * Метод для построения графика с использованием XChart.
     *
     * @param xData      список значений по оси X (например, номер эпохи)
     * @param yData1     список значений для первой серии данных
     * @param yData2     список значений для второй серии данных (может быть null)
     * @param chartTitle заголовок графика
     * @param xAxisTitle название оси X
     * @param yAxisTitle название оси Y
     */
    private static void plotChart(List<Integer> xData, List<Double> yData1, List<Double> yData2,
                                  String chartTitle, String xAxisTitle, String yAxisTitle) {
        XYChart chart = new XYChartBuilder()
                .width(600)
                .height(400)
                .title(chartTitle)
                .xAxisTitle(xAxisTitle)
                .yAxisTitle(yAxisTitle)
                .build();

        chart.addSeries("Серия 1", xData, yData1);
        if (yData2 != null) {
            chart.addSeries("Серия 2", xData, yData2);
        }
        new SwingWrapper<>(chart).displayChart();
    }
}
