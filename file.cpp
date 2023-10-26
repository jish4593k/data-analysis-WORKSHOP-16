#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <algorithm>
#include <bitset>
#include <random>

// Libraries for Machine Learning
#include <armadillo>  // Install Armadillo for data manipulation
#include <mlpack/core.hpp>  // Install mlpack for machine learning

using namespace std;
using namespace arma;
using namespace mlpack;

int main()
{
    // Data Loading
    data::Load("train.csv", train_data, true, false);
    data::Load("test.csv", test_data, true, false);

    cout << "Train Data Info:" << endl;
    cout << train_data.n_rows << " rows and " << train_data.n_cols << " columns." << endl;

    // Data Cleaning
    mat train_features = train_data.submat(span::all, {0, 6});
    vec train_labels = train_data.col(7);

    mat test_features = test_data.submat(span::all, {0, 6});

    // Decision Tree Construction
    mlpack::tree::DecisionTree<> decisionTree(train_features, train_labels, 2, true);

    vec pred_labels;
    decisionTree.Classify(test_features, pred_labels);

    // Calculate Decision Tree Accuracy
    int correct = 0;
    for (size_t i = 0; i < pred_labels.n_elem; ++i)
    {
        if (pred_labels[i] == test_labels[i])
            correct++;
    }
    double accuracy_decision_tree = double(correct) / double(test_labels.n_elem);

    cout << "Decision Tree Accuracy: " << accuracy_decision_tree << endl;

    // Data Visualization (Example: Age Distribution)
    // Data visualization in C++ can be complex; you can use Python libraries for this.

    // Define and compile a simple neural network model
    mlpack::ann::FFN<> model;
    model.Add<mlpack::ann::Linear<>>(train_features.n_cols, 64);
    model.Add<mlpack::ann::ReLULayer<>>();
    model.Add<mlpack::ann::Linear<>>(64, 1);
    model.Add<mlpack::ann::SigmoidLayer<>>();

    mlpack::ann::Adam optimizer(0.001, 32, 0.9, 0.999, 1e-8, train_features.n_rows, 10000);

    // Train the model
    model.Train(train_features, train_labels, optimizer, mlpack::ann::MSE(), 10);

    // Define a simple neural network using PyTorch
    torch::nn::Sequential model;
    model->push_back(torch::nn::Linear(train_features_scaled.size(1), 64));
    model->push_back(torch::nn::ReLU());
    model->push_back(torch::nn::Linear(64, 1));
    model->push_back(torch::nn::Sigmoid());

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.001));

    // Create a DataLoader and train the PyTorch model
    torch::data::datasets::TensorDataset<> train_dataset(train_features_tensor, train_labels_tensor);
    torch::data::DataLoader train_loader(train_dataset, 32, true);

    for (int epoch = 0; epoch < 10; ++epoch)
    {
        model->train();

        double running_loss = 0.0;
        for (auto& batch : *train_loader)
        {
            optimizer.zero_grad();
            auto data = batch.data.to(device);
            auto targets = batch.target.to(device);
            auto outputs = model->forward(data);
            auto loss = torch::binary_cross_entropy(outputs, targets);
            loss.backward();
            optimizer.step();
            running_loss += loss.item<double>();
        }

        cout << "Epoch " << epoch + 1 << ", Loss: " << running_loss / (train_loader.size()) << endl;
    }

    // Evaluate the PyTorch model
    model->eval();

    torch::NoGradGuard no_grad;

    torch::Tensor outputs = model->forward(test_features_scaled_tensor);
    torch::Tensor predicted = (outputs > 0.5).to(torch::kCPU);

    vector<int> predicted_vector(predicted.data<int>(), predicted.data<int>() + predicted.numel());
    // Convert predictions to integers (0 or 1)
    vector<int> predicted = predicted_vector;
    test_data['Survived'] = predicted;
    test_data[['PassengerId', 'Survived']].to_csv("titanic_predictions.csv", index=False);

    return 0;
}
