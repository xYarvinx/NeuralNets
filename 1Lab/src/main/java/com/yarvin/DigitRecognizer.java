package com.yarvin;

import com.yarvin.neuralNetwork.SimpleNeuralNetwork;
import com.yarvin.util.ModelUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class DigitRecognizer extends JFrame {

    private BufferedImage image;
    private JLabel predictionLabel;
    private MultiLayerNetwork model;
    private static Logger logger = LoggerFactory.getLogger(DigitRecognizer.class);

    public DigitRecognizer() {
        setTitle("Рисуйте цифру");
        setSize(600, 650);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);

        try {
            model = ModelSerializer.restoreMultiLayerNetwork(new File("model_28x28_2.zip"));
        } catch (IOException e) {
            e.printStackTrace();
            JOptionPane.showMessageDialog(this, "Ошибка загрузки модели!", "Ошибка", JOptionPane.ERROR_MESSAGE);
            System.exit(1);
        }

        JPanel drawPanel = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                g.drawImage(image, 0, 0, this);
            }
        };
        drawPanel.setPreferredSize(new Dimension(560, 560));
        drawPanel.setBackground(Color.WHITE);

        drawPanel.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                image = new BufferedImage(560, 560, BufferedImage.TYPE_INT_RGB);
                Graphics2D g2d = image.createGraphics();
                g2d.setColor(Color.WHITE);
                g2d.fillRect(0, 0, 560, 560);
                g2d.setColor(Color.BLACK);
                drawPanel.repaint();
            }
        });

        drawPanel.addMouseMotionListener(new MouseAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                Graphics2D g2d = image.createGraphics();
                g2d.setColor(Color.BLACK);
                g2d.fillOval(e.getX() - 10, e.getY() - 10, 20, 20);
                drawPanel.repaint();
            }
        });

        JButton recognizeButton = new JButton("Распознать");
        recognizeButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (image != null) {
                    BufferedImage resizedImage = ModelUtils.resizeImage(image, 28, 28);
                    double[] data = ModelUtils.imageToArray(resizedImage);
                    INDArray input = Nd4j.create(data).reshape(1, 784);
                    INDArray output = model.output(input);
                    int predictedDigit = Nd4j.argMax(output, 1).getInt(0);
                    predictionLabel.setText("Распознанная цифра: " + predictedDigit);
                    logger.info("Распознанная цифра: " + predictedDigit);
                }
            }
        });

        JButton clearButton = new JButton("Очистить");
        clearButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                image = new BufferedImage(560, 560, BufferedImage.TYPE_INT_RGB);
                Graphics2D g2d = image.createGraphics();
                g2d.setColor(Color.WHITE);
                g2d.fillRect(0, 0, 560, 560);
                drawPanel.repaint();
                predictionLabel.setText("Распознанная цифра: ");
                logger.info("Очищено");
            }
        });

        predictionLabel = new JLabel("Распознанная цифра: ", SwingConstants.CENTER);

        JPanel buttonPanel = new JPanel();
        buttonPanel.setLayout(new FlowLayout());
        buttonPanel.add(recognizeButton);
        buttonPanel.add(clearButton);

        setLayout(new BorderLayout());
        add(drawPanel, BorderLayout.CENTER);
        add(buttonPanel, BorderLayout.SOUTH);
        add(predictionLabel, BorderLayout.NORTH);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            DigitRecognizer frame = new DigitRecognizer();
            frame.setVisible(true);
        });
    }
}