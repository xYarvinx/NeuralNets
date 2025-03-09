package com.yarvin.util;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;

public class ModelUtils {

    public static BufferedImage resizeImage(BufferedImage originalImage, int newWidth, int newHeight) {
        BufferedImage resizedImage = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = resizedImage.createGraphics();
        g2d.drawImage(originalImage.getScaledInstance(newWidth, newHeight, Image.SCALE_SMOOTH), 0, 0, null);
        g2d.dispose();
        return resizedImage;
    }

    public static double[] imageToArray(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        double[] data = new double[width * height];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = image.getRGB(x, y);
                int gray = (int) (0.299 * ((rgb >> 16) & 0xFF) + 0.587 * ((rgb >> 8) & 0xFF) + 0.114 * (rgb & 0xFF));

                data[y * width + x] = (255 - gray) / 255.0;
            }
        }
        return data;
    }

    public static DataSet resizeDataSet(DataSet dataSet, int newWidth, int newHeight) {
        INDArray features = dataSet.getFeatures();
        INDArray labels = dataSet.getLabels();
        INDArray resizedFeatures = Nd4j.create(features.size(0), newWidth * newHeight);

        for (int i = 0; i < features.size(0); i++) {
            INDArray image = features.getRow(i).reshape(28, 28);
            BufferedImage bufferedImage = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
            for (int y = 0; y < 28; y++) {
                for (int x = 0; x < 28; x++) {
                    int pixelValue = (int) (image.getDouble(y, x) * 255);
                    bufferedImage.setRGB(x, y, (pixelValue << 16) | (pixelValue << 8) | pixelValue);
                }
            }

            BufferedImage resizedImage = resizeImage(bufferedImage, newWidth, newHeight);
            for (int y = 0; y < newHeight; y++) {
                for (int x = 0; x < newWidth; x++) {
                    int rgb = resizedImage.getRGB(x, y);
                    int gray = (rgb & 0xFF);
                    resizedFeatures.putScalar(i, y * newWidth + x, gray / 255.0);
                }
            }
        }

        return new DataSet(resizedFeatures, labels);
    }
}