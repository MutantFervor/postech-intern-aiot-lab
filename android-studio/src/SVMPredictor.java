package com.example.activity_sensor_testing;

import android.content.Context;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;

public class SVMPredictor {
    private final String modelFilePath;

    public SVMPredictor(Context context, String assetFileName) {
        // Copy the model file from assets to the app's internal storage
        this.modelFilePath = copyModelFileToInternalStorage(context, assetFileName);
    }

    private String copyModelFileToInternalStorage(Context context, String assetFileName) {
        File modelFile = new File(context.getFilesDir(), assetFileName);
        if (!modelFile.exists()) {
            try (InputStream inputStream = context.getAssets().open(assetFileName);
                 FileOutputStream outputStream = new FileOutputStream(modelFile)) {

                byte[] buffer = new byte[1024];
                int length;
                while ((length = inputStream.read(buffer)) > 0) {
                    outputStream.write(buffer, 0, length);
                }
            } catch (IOException e) {
                e.printStackTrace();
                throw new RuntimeException("Error copying model file to internal storage");
            }
        }
        return modelFile.getAbsolutePath();
    }

    public double predict(double[] features) {
        try {
            // Load the SVM model
            svm_model model = svm.svm_load_model(modelFilePath);

            // Create SVM nodes from the features
            svm_node[] nodes = new svm_node[features.length];
            for (int i = 0; i < features.length; i++) {
                svm_node node = new svm_node();
                node.index = i + 1; // SVM node index starts from 1
                node.value = features[i];
                nodes[i] = node;
            }

            // Perform prediction
            double prediction = svm.svm_predict(model, nodes);

            System.out.println("Prediction result: " + prediction);
            return prediction; // This will return the predicted class label
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Error loading SVM model for prediction");
        }
    }

    public double getDecisionValueConfidence(double[] features) {
        try {
            // Load the SVM model
            svm_model model = svm.svm_load_model(modelFilePath);

            // Create SVM nodes from the features
            svm_node[] nodes = new svm_node[features.length];
            for (int i = 0; i < features.length; i++) {
                svm_node node = new svm_node();
                node.index = i + 1; // SVM node index starts from 1
                node.value = features[i];
                nodes[i] = node;
            }

            // Perform prediction and get decision values
            int[] labels = new int[model.nr_class];
            svm.svm_get_labels(model, labels);

            double[] decisionValues = new double[model.nr_class];
            double prediction = svm.svm_predict_values(model, nodes, decisionValues);

            // Calculate confidence as the margin (difference between decision values)
            double maxDecisionValue = Double.NEGATIVE_INFINITY;
            double secondMaxDecisionValue = Double.NEGATIVE_INFINITY;
            for (double value : decisionValues) {
                if (value > maxDecisionValue) {
                    secondMaxDecisionValue = maxDecisionValue;
                    maxDecisionValue = value;
                } else if (value > secondMaxDecisionValue) {
                    secondMaxDecisionValue = value;
                }
            }

            double confidence = maxDecisionValue - secondMaxDecisionValue;
            System.out.println("Prediction confidence (margin): " + confidence);
            return confidence;

        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Error loading SVM model for decision value calculation");
        }
    }


}
