# How to train using AutoML in Azure ML

This project shows how to train a Fashion MNIST model using AutoML, and how to deploy it using an online managed endpoint. It uses MLflow for tracking and model representation.

## Blog post

To learn more about the code in this repo, check out the accompanying blog post: https://bea.stollnitz.com/blog/aml-automl-classification/

## Setup

- You need to have an Azure subscription. You can get a [free subscription](https://azure.microsoft.com/en-us/free) to try it out.
- Create a [resource group](https://docs.microsoft.com/en-us/azure/azure-resource-manager/management/manage-resource-groups-portal).
- Create a new machine learning workspace by following the "Create the workspace" section of the [documentation](https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources). Keep in mind that you'll be creating a "machine learning workspace" Azure resource, not a "workspace" Azure resource, which is entirely different!
- Install the Azure CLI by following the instructions in the [documentation](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli).
- Install the ML extension to the Azure CLI by following the "Installation" section of the [documentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli).
- Install and activate the conda environment by executing the following commands:

```
conda env create -f environment.yml
conda activate aml_automl_classification
```

- Within VS Code, go to the Command Palette clicking "Ctrl + Shift + P," type "Python: Select Interpreter," and select the environment that matches the name of this project.
- In a terminal window, log in to Azure by executing `az login --use-device-code`.
- Set your default subscription by executing `az account set -s "<YOUR_SUBSCRIPTION_NAME_OR_ID>"`. You can verify your default subscription by executing `az account show`, or by looking at `~/.azure/azureProfile.json`.
- Set your default resource group and workspace by executing `az configure --defaults group="<YOUR_RESOURCE_GROUP>" workspace="<YOUR_WORKSPACE>"`. You can verify your defaults by executing `az configure --list-defaults` or by looking at `~/.azure/config`.
- You can now open the [Azure Machine Learning studio](https://ml.azure.com/), where you'll be able to see and manage all the machine learning resources we'll be creating.
- Install the [Azure Machine Learning extension for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.vscode-ai), and log in to it by clicking on "Azure" in the left-hand menu, and then clicking on "Sign in to Azure."

## Train and deploy in the cloud

```
cd aml_automl_classification
```

Under "Run and Debug" on VS Code's left navigation, choose the "Generate data" run configuration and press F5.
Two folders will be created in this project: `automl_test_data` and `automl_train_data`.

Create the compute cluster.

```
az ml compute create -f cloud/cluster-cpu.yml
```

Create the datasets.

```
az ml data create -f cloud/data-test.yml
az ml data create -f cloud/data-train.yml
```

Create the automl job.

```
run_id=$(az ml job create -f cloud/automl-job.yml --query name -o tsv)
```

Go to the Azure ML Studio and wait until the Job completes.
You don't need to download the trained model, but here's how you would do it if you wanted to:

```
az ml job download --name $run_id --output-name "best_model"
```

If you downloaded the model, you can invoke it locally, to make sure all works as expected before invoking your endpoint in the cloud:

```
mlflow models predict --model-uri "named-outputs/best_model" --input-path "test_data/images.csv" --content-type csv --env-manager local
```

Create the Azure ML model from the output.

```
az ml model create --name model-automl-classification --version 1 --path "azureml://jobs/$run_id/outputs/best_model" --type mlflow_model
```

Create the endpoint.

```
az ml online-endpoint create -f cloud/endpoint.yml
az ml online-deployment create -f cloud/deployment.yml --all-traffic
```

Invoke the endpoint.

```
az ml online-endpoint invoke --name endpoint-automl-classification --request-file test_data/images_azureml.json
```

## Related resources

- [What is automated machine learning?](https://learn.microsoft.com/en-us/azure/machine-learning/concept-automated-ml?WT.mc_id=aiml-67318-bstollnitz)
- [AutoML training](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train?WT.mc_id=aiml-67318-bstollnitz)
