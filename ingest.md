Directory structure:
└── graviraja-mlops-basics/
    ├── README.md
    ├── LICENSE
    ├── .dvcignore
    ├── images/
    ├── week_0_project_setup/
    │   ├── README.md
    │   ├── data.py
    │   ├── inference.py
    │   ├── model.py
    │   ├── requirements.txt
    │   ├── train.py
    │   └── experimental_notebooks/
    │       └── data_exploration.ipynb
    ├── week_1_wandb_logging/
    │   ├── README.md
    │   ├── data.py
    │   ├── inference.py
    │   ├── model.py
    │   ├── requirements.txt
    │   ├── train.py
    │   └── experimental_notebooks/
    │       └── data_exploration.ipynb
    ├── week_2_hydra_config/
    │   ├── README.md
    │   ├── data.py
    │   ├── inference.py
    │   ├── model.py
    │   ├── requirements.txt
    │   ├── train.py
    │   ├── configs/
    │   │   ├── config.yaml
    │   │   ├── model/
    │   │   │   └── default.yaml
    │   │   ├── processing/
    │   │   │   └── default.yaml
    │   │   └── training/
    │   │       └── default.yaml
    │   └── experimental_notebooks/
    │       └── data_exploration.ipynb
    ├── week_3_dvc/
    │   ├── README.md
    │   ├── data.py
    │   ├── inference.py
    │   ├── model.py
    │   ├── requirements.txt
    │   ├── train.py
    │   ├── configs/
    │   │   ├── config.yaml
    │   │   ├── model/
    │   │   │   └── default.yaml
    │   │   ├── processing/
    │   │   │   └── default.yaml
    │   │   └── training/
    │   │       └── default.yaml
    │   ├── dvcfiles/
    │   │   └── trained_model.dvc
    │   ├── experimental_notebooks/
    │   │   └── data_exploration.ipynb
    │   └── models/
    │       └── .gitignore
    ├── week_4_onnx/
    │   ├── README.md
    │   ├── convert_model_to_onnx.py
    │   ├── data.py
    │   ├── inference.py
    │   ├── inference_onnx.py
    │   ├── model.py
    │   ├── requirements.txt
    │   ├── train.py
    │   ├── utils.py
    │   ├── configs/
    │   │   ├── config.yaml
    │   │   ├── model/
    │   │   │   └── default.yaml
    │   │   ├── processing/
    │   │   │   └── default.yaml
    │   │   └── training/
    │   │       └── default.yaml
    │   ├── dvcfiles/
    │   │   └── trained_model.dvc
    │   └── experimental_notebooks/
    │       └── data_exploration.ipynb
    ├── week_5_docker/
    │   ├── README.md
    │   ├── app.py
    │   ├── convert_model_to_onnx.py
    │   ├── data.py
    │   ├── docker-compose.yml
    │   ├── Dockerfile
    │   ├── inference.py
    │   ├── inference_onnx.py
    │   ├── model.py
    │   ├── requirements.txt
    │   ├── requirements_inference.txt
    │   ├── train.py
    │   ├── utils.py
    │   ├── configs/
    │   │   ├── config.yaml
    │   │   ├── model/
    │   │   │   └── default.yaml
    │   │   ├── processing/
    │   │   │   └── default.yaml
    │   │   └── training/
    │   │       └── default.yaml
    │   ├── dvcfiles/
    │   │   └── trained_model.dvc
    │   └── experimental_notebooks/
    │       └── data_exploration.ipynb
    ├── week_6_github_actions/
    │   ├── README.md
    │   ├── app.py
    │   ├── convert_model_to_onnx.py
    │   ├── data.py
    │   ├── docker-compose.yml
    │   ├── Dockerfile
    │   ├── inference.py
    │   ├── inference_onnx.py
    │   ├── model.py
    │   ├── parse_json.py
    │   ├── requirements.txt
    │   ├── requirements_inference.txt
    │   ├── train.py
    │   ├── utils.py
    │   ├── configs/
    │   │   ├── config.yaml
    │   │   ├── model/
    │   │   │   └── default.yaml
    │   │   ├── processing/
    │   │   │   └── default.yaml
    │   │   └── training/
    │   │       └── default.yaml
    │   ├── dvcfiles/
    │   │   └── trained_model.dvc
    │   └── experimental_notebooks/
    │       └── data_exploration.ipynb
    ├── week_7_ecr/
    │   ├── README.md
    │   ├── app.py
    │   ├── convert_model_to_onnx.py
    │   ├── data.py
    │   ├── docker-compose.yml
    │   ├── Dockerfile
    │   ├── inference.py
    │   ├── inference_onnx.py
    │   ├── model.py
    │   ├── parse_json.py
    │   ├── requirements.txt
    │   ├── requirements_inference.txt
    │   ├── train.py
    │   ├── utils.py
    │   ├── configs/
    │   │   ├── config.yaml
    │   │   ├── model/
    │   │   │   └── default.yaml
    │   │   ├── processing/
    │   │   │   └── default.yaml
    │   │   └── training/
    │   │       └── default.yaml
    │   ├── dvcfiles/
    │   │   └── trained_model.dvc
    │   └── experimental_notebooks/
    │       └── data_exploration.ipynb
    ├── week_8_serverless/
    │   ├── README.md
    │   ├── app.py
    │   ├── convert_model_to_onnx.py
    │   ├── data.py
    │   ├── docker-compose.yml
    │   ├── Dockerfile
    │   ├── inference.py
    │   ├── inference_onnx.py
    │   ├── lambda_handler.py
    │   ├── model.py
    │   ├── parse_json.py
    │   ├── requirements.txt
    │   ├── requirements_inference.txt
    │   ├── train.py
    │   ├── utils.py
    │   ├── configs/
    │   │   ├── config.yaml
    │   │   ├── model/
    │   │   │   └── default.yaml
    │   │   ├── processing/
    │   │   │   └── default.yaml
    │   │   └── training/
    │   │       └── default.yaml
    │   ├── dvcfiles/
    │   │   └── trained_model.dvc
    │   └── experimental_notebooks/
    │       └── data_exploration.ipynb
    ├── week_9_monitoring/
    │   ├── README.md
    │   ├── app.py
    │   ├── convert_model_to_onnx.py
    │   ├── data.py
    │   ├── docker-compose.yml
    │   ├── Dockerfile
    │   ├── inference.py
    │   ├── inference_onnx.py
    │   ├── lambda_handler.py
    │   ├── model.py
    │   ├── parse_json.py
    │   ├── requirements.txt
    │   ├── requirements_inference.txt
    │   ├── train.py
    │   ├── utils.py
    │   ├── configs/
    │   │   ├── config.yaml
    │   │   ├── model/
    │   │   │   └── default.yaml
    │   │   ├── processing/
    │   │   │   └── default.yaml
    │   │   └── training/
    │   │       └── default.yaml
    │   ├── dvcfiles/
    │   │   └── trained_model.dvc
    │   └── experimental_notebooks/
    │       └── data_exploration.ipynb
    ├── .dvc/
    │   ├── config
    │   ├── .gitignore
    │   └── plots/
    │       ├── confusion.json
    │       ├── confusion_normalized.json
    │       ├── default.json
    │       ├── linear.json
    │       ├── scatter.json
    │       └── smooth.json
    └── .github/
        └── workflows/
            ├── basic.yaml
            └── build_docker_image.yaml


Files Content:

(Files content cropped to 300k characters, download full ingest to see more)
================================================
FILE: README.md
================================================
# MLOps-Basics

 > There is nothing magic about magic. The magician merely understands something simple which doesn’t appear to be simple or natural to the untrained audience. Once you learn how to hold a card while making your hand look empty, you only need practice before you, too, can “do magic.” – Jeffrey Friedl in the book Mastering Regular Expressions

**Note: Please raise an issue for any suggestions, corrections, and feedback.**

The goal of the series is to understand the basics of MLOps like model building, monitoring, configurations, testing, packaging, deployment, cicd, etc.

![pl](images/summary.png)

## Week 0: Project Setup

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>

Refer to the [Blog Post here](https://deep-learning-blogs.vercel.app/blog/mlops-project-setup-part1)

The project I have implemented is a simple classification problem. The scope of this week is to understand the following topics:

- `How to get the data?`
- `How to process the data?`
- `How to define dataloaders?`
- `How to declare the model?`
- `How to train the model?`
- `How to do the inference?`

![pl](images/pl.jpeg)

Following tech stack is used:

- [Huggingface Datasets](https://github.com/huggingface/datasets)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [Pytorch Lightning](https://pytorch-lightning.readthedocs.io/)

## Week 1: Model monitoring - Weights and Biases

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>

Refer to the [Blog Post here](https://deep-learning-blogs.vercel.app/blog/mlops-wandb-integration)

Tracking all the experiments like tweaking hyper-parameters, trying different models to test their performance and seeing the connection between model and the input data will help in developing a better model.

The scope of this week is to understand the following topics:

- `How to configure basic logging with W&B?`
- `How to compute metrics and log them in W&B?`
- `How to add plots in W&B?`
- `How to add data samples to W&B?`

![wannb](images/wandb.png)

Following tech stack is used:

- [Weights and Biases](https://wandb.ai/site)
- [torchmetrics](https://torchmetrics.readthedocs.io/)

References:

- [Tutorial on Pytorch Lightning + Weights & Bias](https://www.youtube.com/watch?v=hUXQm46TAKc)

- [WandB Documentation](https://docs.wandb.ai/)

## Week 2: Configurations - Hydra

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>

Refer to the [Blog Post here](https://deep-learning-blogs.vercel.app/blog/mlops-hydra-config)

Configuration management is a necessary for managing complex software systems. Lack of configuration management can cause serious problems with reliability, uptime, and the ability to scale a system.

The scope of this week is to understand the following topics:

- `Basics of Hydra`
- `Overridding configurations`
- `Splitting configuration across multiple files`
- `Variable Interpolation`
- `How to run model with different parameter combinations?`

![hydra](images/hydra.png)

Following tech stack is used:

- [Hydra](https://hydra.cc/)

References

- [Hydra Documentation](https://hydra.cc/docs/intro)

- [Simone Tutorial on Hydra](https://www.sscardapane.it/tutorials/hydra-tutorial/#executing-multiple-runs)


## Week 3: Data Version Control - DVC

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>

Refer to the [Blog Post here](https://deep-learning-blogs.vercel.app/blog/mlops-dvc)

Classical code version control systems are not designed to handle large files, which make cloning and storing the history impractical. Which are very common in Machine Learning.

The scope of this week is to understand the following topics:

- `Basics of DVC`
- `Initialising DVC`
- `Configuring Remote Storage`
- `Saving Model to the Remote Storage`
- `Versioning the models`

![dvc](images/dvc.png)

Following tech stack is used:

- [DVC](https://dvc.org/)

References

- [DVC Documentation](https://dvc.org/doc)

- [DVC Tutorial on Versioning data](https://www.youtube.com/watch?v=kLKBcPonMYw)

## Week 4: Model Packaging - ONNX

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>

Refer to the [Blog Post here](https://deep-learning-blogs.vercel.app/blog/mlops-onnx)

Why do we need model packaging? Models can be built using any machine learning framework available out there (sklearn, tensorflow, pytorch, etc.). We might want to deploy models in different environments like (mobile, web, raspberry pi) or want to run in a different framework (trained in pytorch, inference in tensorflow).
A common file format to enable AI developers to use models with a variety of frameworks, tools, runtimes, and compilers will help a lot.

This is acheived by a community project `ONNX`.

The scope of this week is to understand the following topics:

- `What is ONNX?`

- `How to convert a trained model to ONNX format?`

- `What is ONNX Runtime?`

- `How to run ONNX converted model in ONNX Runtime?`

- `Comparisions`

![ONNX](images/onnx.jpeg)

Following tech stack is used:

- [ONNX](https://onnx.ai/)
- [ONNXRuntime](https://www.onnxruntime.ai/)

References

- [Abhishek Thakur tutorial on onnx model conversion](https://www.youtube.com/watch?v=7nutT3Aacyw)
- [Pytorch Lightning documentation on onnx conversion](https://pytorch-lightning.readthedocs.io/en/stable/common/production_inference.html)
- [Huggingface Blog on ONNXRuntime](https://medium.com/microsoftazure/accelerate-your-nlp-pipelines-using-hugging-face-transformers-and-onnx-runtime-2443578f4333)
- [Piotr Blog on onnx conversion](https://tugot17.github.io/data-science-blog/onnx/tutorial/2020/09/21/Exporting-lightning-model-to-onnx.html)


## Week 5: Model Packaging - Docker

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>

Refer to the [Blog Post here](https://deep-learning-blogs.vercel.app/blog/mlops-docker)

Why do we need packaging? We might have to share our application with others, and when they try to run the application most of the time it doesn’t run due to dependencies issues / OS related issues and for that, we say (famous quote across engineers) that `It works on my laptop/system`.

So for others to run the applications they have to set up the same environment as it was run on the host side which means a lot of manual configuration and installation of components.

The solution to these limitations is a technology called Containers.

By containerizing/packaging the application, we can run the application on any cloud platform to get advantages of managed services and autoscaling and reliability, and many more.

The most prominent tool to do the packaging of application is Docker 🛳

The scope of this week is to understand the following topics:

- `FastAPI wrapper`
- `Basics of Docker`
- `Building Docker Container`
- `Docker Compose`

![Docker](images/docker_flow.png)

References

- [Analytics vidhya blog](https://www.analyticsvidhya.com/blog/2021/06/a-hands-on-guide-to-containerized-your-machine-learning-workflow-with-docker/)


## Week 6: CI/CD - GitHub Actions

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>

Refer to the [Blog Post here](https://deep-learning-blogs.vercel.app/blog/mlops-github-actions)

CI/CD is a coding philosophy and set of practices with which you can continuously build, test, and deploy iterative code changes.

This iterative process helps reduce the chance that you develop new code based on a buggy or failed previous versions. With this method, you strive to have less human intervention or even no intervention at all, from the development of new code until its deployment.

In this post, I will be going through the following topics:

- Basics of GitHub Actions
- First GitHub Action
- Creating Google Service Account
- Giving access to Service account
- Configuring DVC to use Google Service account
- Configuring Github Action

![Docker](images/basic_flow.png)

References

- [Configuring service account](https://dvc.org/doc/user-guide/setup-google-drive-remote)

- [Github actions](https://docs.github.com/en/actions/quickstart)


## Week 7: Container Registry - AWS ECR

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>

Refer to the [Blog Post here](https://deep-learning-blogs.vercel.app/blog/mlops-container-registry)

A container registry is a place to store container images. A container image is a file comprised of multiple layers which can execute applications in a single instance. Hosting all the images in one stored location allows users to commit, identify and pull images when needed.

Amazon Simple Storage Service (S3) is a storage for the internet. It is designed for large-capacity, low-cost storage provision across multiple geographical regions.

In this week, I will be going through the following topics:

- `Basics of S3`

- `Programmatic access to S3`

- `Configuring AWS S3 as remote storage in DVC`

- `Basics of ECR`

- `Configuring GitHub Actions to use S3, ECR`

![Docker](images/ecr_flow.png)


## Week 8: Serverless Deployment - AWS Lambda

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>

Refer to the [Blog Post here](https://deep-learning-blogs.vercel.app/blog/mlops-serverless)

A serverless architecture is a way to build and run applications and services without having to manage infrastructure. The application still runs on servers, but all the server management is done by third party service (AWS). We no longer have to provision, scale, and maintain servers to run the applications. By using a serverless architecture, developers can focus on their core product instead of worrying about managing and operating servers or runtimes, either in the cloud or on-premises.

In this week, I will be going through the following topics:

- `Basics of Serverless`

- `Basics of AWS Lambda`

- `Triggering Lambda with API Gateway`

- `Deploying Container using Lambda`

- `Automating deployment to Lambda using Github Actions`

![Docker](images/lambda_flow.png)


## Week 9: Prediction Monitoring - Kibana

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>

Refer to the [Blog Post here](https://deep-learning-blogs.vercel.app/blog/mlops-monitoring)


Monitoring systems can help give us confidence that our systems are running smoothly and, in the event of a system failure, can quickly provide appropriate context when diagnosing the root cause.

Things we want to monitor during and training and inference are different. During training we are concered about whether the loss is decreasing or not, whether the model is overfitting, etc.

But, during inference, We like to have confidence that our model is making correct predictions.

There are many reasons why a model can fail to make useful predictions:

- The underlying data distribution has shifted over time and the model has gone stale. i.e inference data characteristics is different from the data characteristics used to train the model.

- The inference data stream contains edge cases (not seen during model training). In this scenarios model might perform poorly or can lead to errors.

- The model was misconfigured in its production deployment. (Configuration issues are common)

In all of these scenarios, the model could still make a `successful` prediction from a service perspective, but the predictions will likely not be useful. Monitoring machine learning models can help us detect such scenarios and intervene (e.g. trigger a model retraining/deployment pipeline).

In this week, I will be going through the following topics:

- `Basics of Cloudwatch Logs`

- `Creating Elastic Search Cluster`

- `Configuring Cloudwatch Logs with Elastic Search`

- `Creating Index Patterns in Kibana`

- `Creating Kibana Visualisations`

- `Creating Kibana Dashboard`

![Docker](images/kibana_flow.png)



================================================
FILE: LICENSE
================================================
MIT License

Copyright (c) 2021 raviraja

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



================================================
FILE: .dvcignore
================================================
# Add patterns of files dvc should ignore, which could improve
# the performance. Learn more at
# https://dvc.org/doc/user-guide/dvcignore




================================================
FILE: week_0_project_setup/README.md
================================================

**Note: The purpose of the project to explore the libraries and learn how to use them. Not to build a SOTA model.**

## Requirements:

This project uses Python 3.8

Create a virtual env with the following command:

```
conda create --name project-setup python=3.8
conda activate project-setup
```

Install the requirements:

```
pip install -r requirements.txt
```

## Running

### Training

After installing the requirements, in order to train the model simply run:

```
python train.py
```

### Inference

After training, update the model checkpoint path in the code and run

```
python inference.py
```

### Running notebooks

I am using [Jupyter lab](https://jupyter.org/install) to run the notebooks. 

Since I am using a virtualenv, when I run the command `jupyter lab` it might or might not use the virtualenv.

To make sure to use the virutalenv, run the following commands before running `jupyter lab`

```
conda install ipykernel
python -m ipykernel install --user --name project-setup
pip install ipywidgets
```





================================================
FILE: week_0_project_setup/data.py
================================================
import torch
import datasets
import pytorch_lightning as pl

from datasets import load_dataset
from transformers import AutoTokenizer


class DataModule(pl.LightningDataModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", batch_size=32):
        super().__init__()

        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self):
        cola_dataset = load_dataset("glue", "cola")
        self.train_data = cola_dataset["train"]
        self.val_data = cola_dataset["validation"]

    def tokenize_data(self, example):
        return self.tokenizer(
            example["sentence"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

    def setup(self, stage=None):
        # we set up only relevant datasets when stage is specified
        if stage == "fit" or stage is None:
            self.train_data = self.train_data.map(self.tokenize_data, batched=True)
            self.train_data.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"]
            )

            self.val_data = self.val_data.map(self.tokenize_data, batched=True)
            self.val_data.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"]
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False
        )


if __name__ == "__main__":
    data_model = DataModule()
    data_model.prepare_data()
    data_model.setup()
    print(next(iter(data_model.train_dataloader()))["input_ids"].shape)



================================================
FILE: week_0_project_setup/inference.py
================================================
import torch
from model import ColaModel
from data import DataModule


class ColaPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = ColaModel.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.freeze()
        self.processor = DataModule()
        self.softmax = torch.nn.Softmax(dim=0)
        self.lables = ["unacceptable", "acceptable"]

    def predict(self, text):
        inference_sample = {"sentence": text}
        processed = self.processor.tokenize_data(inference_sample)
        logits = self.model(
            torch.tensor([processed["input_ids"]]),
            torch.tensor([processed["attention_mask"]]),
        )
        scores = self.softmax(logits[0]).tolist()
        predictions = []
        for score, label in zip(scores, self.lables):
            predictions.append({"label": label, "score": score})
        return predictions


if __name__ == "__main__":
    sentence = "The boy is sitting on a bench"
    predictor = ColaPredictor("./models/epoch=0-step=267.ckpt")
    print(predictor.predict(sentence))



================================================
FILE: week_0_project_setup/model.py
================================================
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from transformers import AutoModel
from sklearn.metrics import accuracy_score


class ColaModel(pl.LightningModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", lr=1e-2):
        super(ColaModel, self).__init__()
        self.save_hyperparameters()

        self.bert = AutoModel.from_pretrained(model_name)
        self.W = nn.Linear(self.bert.config.hidden_size, 2)
        self.num_classes = 2

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        h_cls = outputs.last_hidden_state[:, 0]
        logits = self.W(h_cls)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["label"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["label"])
        _, preds = torch.max(logits, dim=1)
        val_acc = accuracy_score(preds.cpu(), batch["label"].cpu())
        val_acc = torch.tensor(val_acc)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])



================================================
FILE: week_0_project_setup/requirements.txt
================================================
pytorch-lightning==1.2.10
datasets==1.6.2
transformers==4.5.1
scikit-learn==0.24.2


================================================
FILE: week_0_project_setup/train.py
================================================
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from data import DataModule
from model import ColaModel


def main():
    cola_data = DataModule()
    cola_model = ColaModel()

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="val_loss", mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
    )

    trainer = pl.Trainer(
        default_root_dir="logs",
        gpus=(1 if torch.cuda.is_available() else 0),
        max_epochs=5,
        fast_dev_run=False,
        logger=pl.loggers.TensorBoardLogger("logs/", name="cola", version=1),
        callbacks=[checkpoint_callback, early_stopping_callback],
    )
    trainer.fit(cola_model, cola_data)


if __name__ == "__main__":
    main()



================================================
FILE: week_0_project_setup/experimental_notebooks/data_exploration.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
## Imports
"""

import datasets
import pandas as pd

from datasets import load_dataset

"""
## Dataset
"""

cola_dataset = load_dataset('glue', 'cola')
# Output:
#   Reusing dataset glue (/Users/raviraja/.cache/huggingface/datasets/glue/cola/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad)


cola_dataset
# Output:
#   DatasetDict({

#       train: Dataset({

#           features: ['sentence', 'label', 'idx'],

#           num_rows: 8551

#       })

#       validation: Dataset({

#           features: ['sentence', 'label', 'idx'],

#           num_rows: 1043

#       })

#       test: Dataset({

#           features: ['sentence', 'label', 'idx'],

#           num_rows: 1063

#       })

#   })

train_dataset = cola_dataset['train']
val_dataset = cola_dataset['validation']
test_dataset = cola_dataset['test']

len(train_dataset), len(val_dataset), len(test_dataset)
# Output:
#   (8551, 1043, 1063)

train_dataset[0]
# Output:
#   {'idx': 0,

#    'label': 1,

#    'sentence': "Our friends won't buy this analysis, let alone the next one we propose."}

val_dataset[0]
# Output:
#   {'idx': 0,

#    'label': 1,

#    'sentence': 'The sailors rode the breeze clear of the rocks.'}

test_dataset[0]
# Output:
#   {'idx': 0, 'label': -1, 'sentence': 'Bill whistled past the house.'}

train_dataset.features
# Output:
#   {'sentence': Value(dtype='string', id=None),

#    'label': ClassLabel(num_classes=2, names=['unacceptable', 'acceptable'], names_file=None, id=None),

#    'idx': Value(dtype='int32', id=None)}

train_dataset.filter(lambda example: example['label'] == train_dataset.features['label'].str2int('acceptable'))[:5]
# Output:
#   HBox(children=(FloatProgress(value=0.0, max=9.0), HTML(value='')))
#   

#   {'idx': [0, 1, 2, 3, 4],

#    'label': [1, 1, 1, 1, 1],

#    'sentence': ["Our friends won't buy this analysis, let alone the next one we propose.",

#     "One more pseudo generalization and I'm giving up.",

#     "One more pseudo generalization or I'm giving up.",

#     'The more we study verbs, the crazier they get.',

#     'Day by day the facts are getting murkier.']}

train_dataset.filter(lambda example: example['label'] == train_dataset.features['label'].str2int('unacceptable'))[:5]
# Output:
#   HBox(children=(FloatProgress(value=0.0, max=9.0), HTML(value='')))
#   

#   {'idx': [18, 20, 22, 23, 25],

#    'label': [0, 0, 0, 0, 0],

#    'sentence': ['They drank the pub.',

#     'The professor talked us.',

#     'We yelled ourselves.',

#     'We yelled Harry hoarse.',

#     'Harry coughed himself.']}

"""
## Tokenizing
"""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")

train_dataset = cola_dataset['train']
val_dataset = cola_dataset['validation']
test_dataset = cola_dataset['test']

tokenizer
# Output:
#   PreTrainedTokenizerFast(name_or_path='google/bert_uncased_L-2_H-128_A-2', vocab_size=30522, model_max_len=1000000000000000019884624838656, is_fast=True, padding_side='right', special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'})

print(train_dataset[0]['sentence'])
tokenizer(train_dataset[0]['sentence'])
# Output:
#   Our friends won't buy this analysis, let alone the next one we propose.

#   {'input_ids': [101, 2256, 2814, 2180, 1005, 1056, 4965, 2023, 4106, 1010, 2292, 2894, 1996, 2279, 2028, 2057, 16599, 1012, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

tokenizer.decode(tokenizer(train_dataset[0]['sentence'])['input_ids'])
# Output:
#   "[CLS] our friends won't buy this analysis, let alone the next one we propose. [SEP]"

def encode(examples):
    return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

train_dataset = train_dataset.map(encode, batched=True)
# Output:
#   HBox(children=(FloatProgress(value=0.0, max=9.0), HTML(value='')))
#   


"""
## Formatting
"""

import torch

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

"""
## Data Loader
"""

dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

next(iter(dataloader))
# Output:
#   {'attention_mask': tensor([[1, 1, 1,  ..., 0, 0, 0],

#            [1, 1, 1,  ..., 0, 0, 0],

#            [1, 1, 1,  ..., 0, 0, 0],

#            ...,

#            [1, 1, 1,  ..., 0, 0, 0],

#            [1, 1, 1,  ..., 0, 0, 0],

#            [1, 1, 1,  ..., 0, 0, 0]]),

#    'input_ids': tensor([[  101,  2256,  2814,  ...,     0,     0,     0],

#            [  101,  2028,  2062,  ...,     0,     0,     0],

#            [  101,  2028,  2062,  ...,     0,     0,     0],

#            ...,

#            [  101,  5965, 12808,  ...,     0,     0,     0],

#            [  101,  2198, 10948,  ...,     0,     0,     0],

#            [  101,  3021, 24471,  ...,     0,     0,     0]]),

#    'label': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0,

#            1, 0, 0, 1, 1, 1, 1, 1])}

for batch in dataloader:
    print(batch['input_ids'].shape, batch['attention_mask'].shape, batch['label'].shape)
# Output:
#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([7, 512]) torch.Size([7, 512]) torch.Size([7])




================================================
FILE: week_1_wandb_logging/README.md
================================================

**Note: The purpose of the project to explore the libraries and learn how to use them. Not to build a SOTA model.**

## Requirements:

This project uses Python 3.8

Create a virtual env with the following command:

```
conda create --name project-setup python=3.8
conda activate project-setup
```

Install the requirements:

```
pip install -r requirements.txt
```

## Running

### Training

After installing the requirements, in order to train the model simply run:

```
python train.py
```

### Monitoring

Once the training is completed in the end of the logs you will see something like:

```
wandb: Synced 5 W&B file(s), 4 media file(s), 3 artifact file(s) and 0 other file(s)
wandb: 
wandb: Synced proud-mountain-77: https://wandb.ai/raviraja/MLOps%20Basics/runs/3vp1twdc
```

Follow the link to see the wandb dashboard which contains all the plots.

### Inference

After training, update the model checkpoint path in the code and run

```
python inference.py
```

### Running notebooks

I am using [Jupyter lab](https://jupyter.org/install) to run the notebooks. 

Since I am using a virtualenv, when I run the command `jupyter lab` it might or might not use the virtualenv.

To make sure to use the virutalenv, run the following commands before running `jupyter lab`

```
conda install ipykernel
python -m ipykernel install --user --name project-setup
pip install ipywidgets
```


================================================
FILE: week_1_wandb_logging/data.py
================================================
import torch
import datasets
import pytorch_lightning as pl

from datasets import load_dataset
from transformers import AutoTokenizer


class DataModule(pl.LightningDataModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", batch_size=64):
        super().__init__()

        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self):
        cola_dataset = load_dataset("glue", "cola")
        self.train_data = cola_dataset["train"]
        self.val_data = cola_dataset["validation"]

    def tokenize_data(self, example):
        return self.tokenizer(
            example["sentence"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    def setup(self, stage=None):
        # we set up only relevant datasets when stage is specified
        if stage == "fit" or stage is None:
            self.train_data = self.train_data.map(self.tokenize_data, batched=True)
            self.train_data.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"]
            )

            self.val_data = self.val_data.map(self.tokenize_data, batched=True)
            self.val_data.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "label"],
                output_all_columns=True,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False
        )


if __name__ == "__main__":
    data_model = DataModule()
    data_model.prepare_data()
    data_model.setup()
    print(next(iter(data_model.train_dataloader()))["input_ids"].shape)



================================================
FILE: week_1_wandb_logging/inference.py
================================================
import torch
from model import ColaModel
from data import DataModule


class ColaPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = ColaModel.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.freeze()
        self.processor = DataModule()
        self.softmax = torch.nn.Softmax(dim=0)
        self.lables = ["unacceptable", "acceptable"]

    def predict(self, text):
        inference_sample = {"sentence": text}
        processed = self.processor.tokenize_data(inference_sample)
        logits = self.model(
            torch.tensor([processed["input_ids"]]),
            torch.tensor([processed["attention_mask"]]),
        )
        scores = self.softmax(logits[0]).tolist()
        predictions = []
        for score, label in zip(scores, self.lables):
            predictions.append({"label": label, "score": score})
        return predictions


if __name__ == "__main__":
    sentence = "The boy is sitting on a bench"
    predictor = ColaPredictor("./models/epoch=0-step=267.ckpt")
    print(predictor.predict(sentence))



================================================
FILE: week_1_wandb_logging/model.py
================================================
import torch
import wandb
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification
import torchmetrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class ColaModel(pl.LightningModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", lr=3e-5):
        super(ColaModel, self).__init__()
        self.save_hyperparameters()

        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        self.num_classes = 2
        self.train_accuracy_metric = torchmetrics.Accuracy()
        self.val_accuracy_metric = torchmetrics.Accuracy()
        self.f1_metric = torchmetrics.F1(num_classes=self.num_classes)
        self.precision_macro_metric = torchmetrics.Precision(
            average="macro", num_classes=self.num_classes
        )
        self.recall_macro_metric = torchmetrics.Recall(
            average="macro", num_classes=self.num_classes
        )
        self.precision_micro_metric = torchmetrics.Precision(average="micro")
        self.recall_micro_metric = torchmetrics.Recall(average="micro")

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        # loss = F.cross_entropy(logits, batch["label"])
        preds = torch.argmax(outputs.logits, 1)
        train_acc = self.train_accuracy_metric(preds, batch["label"])
        self.log("train/loss", outputs.loss, prog_bar=True, on_epoch=True)
        self.log("train/acc", train_acc, prog_bar=True, on_epoch=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        labels = batch["label"]
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        preds = torch.argmax(outputs.logits, 1)

        # Metrics
        valid_acc = self.val_accuracy_metric(preds, labels)
        precision_macro = self.precision_macro_metric(preds, labels)
        recall_macro = self.recall_macro_metric(preds, labels)
        precision_micro = self.precision_micro_metric(preds, labels)
        recall_micro = self.recall_micro_metric(preds, labels)
        f1 = self.f1_metric(preds, labels)

        # Logging metrics
        self.log("valid/loss", outputs.loss, prog_bar=True, on_step=True)
        self.log("valid/acc", valid_acc, prog_bar=True, on_epoch=True)
        self.log("valid/precision_macro", precision_macro, prog_bar=True, on_epoch=True)
        self.log("valid/recall_macro", recall_macro, prog_bar=True, on_epoch=True)
        self.log("valid/precision_micro", precision_micro, prog_bar=True, on_epoch=True)
        self.log("valid/recall_micro", recall_micro, prog_bar=True, on_epoch=True)
        self.log("valid/f1", f1, prog_bar=True, on_epoch=True)
        return {"labels": labels, "logits": outputs.logits}

    def validation_epoch_end(self, outputs):
        labels = torch.cat([x["labels"] for x in outputs])
        logits = torch.cat([x["logits"] for x in outputs])
        preds = torch.argmax(logits, 1)

        ## There are multiple ways to track the metrics
        # 1. Confusion matrix plotting using inbuilt W&B method
        self.logger.experiment.log(
            {
                "conf": wandb.plot.confusion_matrix(
                    probs=logits.numpy(), y_true=labels.numpy()
                )
            }
        )

        # 2. Confusion Matrix plotting using scikit-learn method
        # wandb.log({"cm": wandb.sklearn.plot_confusion_matrix(labels.numpy(), preds)})

        # 3. Confusion Matric plotting using Seaborn
        # data = confusion_matrix(labels.numpy(), preds.numpy())
        # df_cm = pd.DataFrame(data, columns=np.unique(labels), index=np.unique(labels))
        # df_cm.index.name = "Actual"
        # df_cm.columns.name = "Predicted"
        # plt.figure(figsize=(7, 4))
        # plot = sns.heatmap(
        #     df_cm, cmap="Blues", annot=True, annot_kws={"size": 16}
        # )  # font size
        # self.logger.experiment.log({"Confusion Matrix": wandb.Image(plot)})

        # self.logger.experiment.log(
        #     {"roc": wandb.plot.roc_curve(labels.numpy(), logits.numpy())}
        # )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])



================================================
FILE: week_1_wandb_logging/requirements.txt
================================================
pytorch-lightning==1.2.10
datasets==1.6.2
transformers==4.5.1
scikit-learn==0.24.2
wandb
torchmetrics
matplotlib
seaborn


================================================
FILE: week_1_wandb_logging/train.py
================================================
import torch
import wandb
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from data import DataModule
from model import ColaModel


class SamplesVisualisationLogger(pl.Callback):
    def __init__(self, datamodule):
        super().__init__()

        self.datamodule = datamodule

    def on_validation_end(self, trainer, pl_module):
        val_batch = next(iter(self.datamodule.val_dataloader()))
        sentences = val_batch["sentence"]

        outputs = pl_module(val_batch["input_ids"], val_batch["attention_mask"])
        preds = torch.argmax(outputs.logits, 1)
        labels = val_batch["label"]

        df = pd.DataFrame(
            {"Sentence": sentences, "Label": labels.numpy(), "Predicted": preds.numpy()}
        )

        wrong_df = df[df["Label"] != df["Predicted"]]
        trainer.logger.experiment.log(
            {
                "examples": wandb.Table(dataframe=wrong_df, allow_mixed_types=True),
                "global_step": trainer.global_step,
            }
        )


def main():
    cola_data = DataModule()
    cola_model = ColaModel()

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models",
        filename="best-checkpoint.ckpt",
        monitor="valid/loss",
        mode="min",
    )

    early_stopping_callback = EarlyStopping(
        monitor="valid/loss", patience=3, verbose=True, mode="min"
    )

    wandb_logger = WandbLogger(project="MLOps Basics", entity="raviraja")
    trainer = pl.Trainer(
        max_epochs=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, SamplesVisualisationLogger(cola_data), early_stopping_callback],
        log_every_n_steps=10,
        deterministic=True,
        # limit_train_batches=0.25,
        # limit_val_batches=0.25
    )
    trainer.fit(cola_model, cola_data)


if __name__ == "__main__":
    main()



================================================
FILE: week_1_wandb_logging/experimental_notebooks/data_exploration.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
## Imports
"""

import datasets
import pandas as pd

from datasets import load_dataset

"""
## Dataset
"""

cola_dataset = load_dataset('glue', 'cola')
# Output:
#   Reusing dataset glue (/Users/raviraja/.cache/huggingface/datasets/glue/cola/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad)


cola_dataset
# Output:
#   DatasetDict({

#       train: Dataset({

#           features: ['sentence', 'label', 'idx'],

#           num_rows: 8551

#       })

#       validation: Dataset({

#           features: ['sentence', 'label', 'idx'],

#           num_rows: 1043

#       })

#       test: Dataset({

#           features: ['sentence', 'label', 'idx'],

#           num_rows: 1063

#       })

#   })

train_dataset = cola_dataset['train']
val_dataset = cola_dataset['validation']
test_dataset = cola_dataset['test']

len(train_dataset), len(val_dataset), len(test_dataset)
# Output:
#   (8551, 1043, 1063)

train_dataset[0]
# Output:
#   {'idx': 0,

#    'label': 1,

#    'sentence': "Our friends won't buy this analysis, let alone the next one we propose."}

val_dataset[0]
# Output:
#   {'idx': 0,

#    'label': 1,

#    'sentence': 'The sailors rode the breeze clear of the rocks.'}

test_dataset[0]
# Output:
#   {'idx': 0, 'label': -1, 'sentence': 'Bill whistled past the house.'}

train_dataset.features
# Output:
#   {'sentence': Value(dtype='string', id=None),

#    'label': ClassLabel(num_classes=2, names=['unacceptable', 'acceptable'], names_file=None, id=None),

#    'idx': Value(dtype='int32', id=None)}

train_dataset.filter(lambda example: example['label'] == train_dataset.features['label'].str2int('acceptable'))[:5]
# Output:
#   HBox(children=(FloatProgress(value=0.0, max=9.0), HTML(value='')))
#   

#   {'idx': [0, 1, 2, 3, 4],

#    'label': [1, 1, 1, 1, 1],

#    'sentence': ["Our friends won't buy this analysis, let alone the next one we propose.",

#     "One more pseudo generalization and I'm giving up.",

#     "One more pseudo generalization or I'm giving up.",

#     'The more we study verbs, the crazier they get.',

#     'Day by day the facts are getting murkier.']}

train_dataset.filter(lambda example: example['label'] == train_dataset.features['label'].str2int('unacceptable'))[:5]
# Output:
#   HBox(children=(FloatProgress(value=0.0, max=9.0), HTML(value='')))
#   

#   {'idx': [18, 20, 22, 23, 25],

#    'label': [0, 0, 0, 0, 0],

#    'sentence': ['They drank the pub.',

#     'The professor talked us.',

#     'We yelled ourselves.',

#     'We yelled Harry hoarse.',

#     'Harry coughed himself.']}

"""
## Tokenizing
"""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")

train_dataset = cola_dataset['train']
val_dataset = cola_dataset['validation']
test_dataset = cola_dataset['test']

tokenizer
# Output:
#   PreTrainedTokenizerFast(name_or_path='google/bert_uncased_L-2_H-128_A-2', vocab_size=30522, model_max_len=1000000000000000019884624838656, is_fast=True, padding_side='right', special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'})

print(train_dataset[0]['sentence'])
tokenizer(train_dataset[0]['sentence'])
# Output:
#   Our friends won't buy this analysis, let alone the next one we propose.

#   {'input_ids': [101, 2256, 2814, 2180, 1005, 1056, 4965, 2023, 4106, 1010, 2292, 2894, 1996, 2279, 2028, 2057, 16599, 1012, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

tokenizer.decode(tokenizer(train_dataset[0]['sentence'])['input_ids'])
# Output:
#   "[CLS] our friends won't buy this analysis, let alone the next one we propose. [SEP]"

def encode(examples):
    return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

train_dataset = train_dataset.map(encode, batched=True)
# Output:
#   HBox(children=(FloatProgress(value=0.0, max=9.0), HTML(value='')))
#   


"""
## Formatting
"""

import torch

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

"""
## Data Loader
"""

dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

next(iter(dataloader))
# Output:
#   {'attention_mask': tensor([[1, 1, 1,  ..., 0, 0, 0],

#            [1, 1, 1,  ..., 0, 0, 0],

#            [1, 1, 1,  ..., 0, 0, 0],

#            ...,

#            [1, 1, 1,  ..., 0, 0, 0],

#            [1, 1, 1,  ..., 0, 0, 0],

#            [1, 1, 1,  ..., 0, 0, 0]]),

#    'input_ids': tensor([[  101,  2256,  2814,  ...,     0,     0,     0],

#            [  101,  2028,  2062,  ...,     0,     0,     0],

#            [  101,  2028,  2062,  ...,     0,     0,     0],

#            ...,

#            [  101,  5965, 12808,  ...,     0,     0,     0],

#            [  101,  2198, 10948,  ...,     0,     0,     0],

#            [  101,  3021, 24471,  ...,     0,     0,     0]]),

#    'label': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0,

#            1, 0, 0, 1, 1, 1, 1, 1])}

for batch in dataloader:
    print(batch['input_ids'].shape, batch['attention_mask'].shape, batch['label'].shape)
# Output:
#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([7, 512]) torch.Size([7, 512]) torch.Size([7])




================================================
FILE: week_2_hydra_config/README.md
================================================

**Note: The purpose of the project to explore the libraries and learn how to use them. Not to build a SOTA model.**

## Requirements:

This project uses Python 3.8

Create a virtual env with the following command:

```
conda create --name project-setup python=3.8
conda activate project-setup
```

Install the requirements:

```
pip install -r requirements.txt
```

## Running

### Training

After installing the requirements, in order to train the model simply run:

```
python train.py
```

### Monitoring

Once the training is completed in the end of the logs you will see something like:

```
wandb: Synced 5 W&B file(s), 4 media file(s), 3 artifact file(s) and 0 other file(s)
wandb: 
wandb: Synced proud-mountain-77: https://wandb.ai/raviraja/MLOps%20Basics/runs/3vp1twdc
```

Follow the link to see the wandb dashboard which contains all the plots.

### Inference

After training, update the model checkpoint path in the code and run

```
python inference.py
```

### Running notebooks

I am using [Jupyter lab](https://jupyter.org/install) to run the notebooks. 

Since I am using a virtualenv, when I run the command `jupyter lab` it might or might not use the virtualenv.

To make sure to use the virutalenv, run the following commands before running `jupyter lab`

```
conda install ipykernel
python -m ipykernel install --user --name project-setup
pip install ipywidgets
```


================================================
FILE: week_2_hydra_config/data.py
================================================
import torch
import datasets
import pytorch_lightning as pl

from datasets import load_dataset
from transformers import AutoTokenizer


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name="google/bert_uncased_L-2_H-128_A-2",
        batch_size=64,
        max_length=128,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self):
        cola_dataset = load_dataset("glue", "cola")
        self.train_data = cola_dataset["train"]
        self.val_data = cola_dataset["validation"]

    def tokenize_data(self, example):
        return self.tokenizer(
            example["sentence"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

    def setup(self, stage=None):
        # we set up only relevant datasets when stage is specified
        if stage == "fit" or stage is None:
            self.train_data = self.train_data.map(self.tokenize_data, batched=True)
            self.train_data.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"]
            )

            self.val_data = self.val_data.map(self.tokenize_data, batched=True)
            self.val_data.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "label"],
                output_all_columns=True,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False
        )


if __name__ == "__main__":
    data_model = DataModule()
    data_model.prepare_data()
    data_model.setup()
    print(next(iter(data_model.train_dataloader()))["input_ids"].shape)



================================================
FILE: week_2_hydra_config/inference.py
================================================
import torch
from model import ColaModel
from data import DataModule


class ColaPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = ColaModel.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.freeze()
        self.processor = DataModule()
        self.softmax = torch.nn.Softmax(dim=0)
        self.lables = ["unacceptable", "acceptable"]

    def predict(self, text):
        inference_sample = {"sentence": text}
        processed = self.processor.tokenize_data(inference_sample)
        logits = self.model(
            torch.tensor([processed["input_ids"]]),
            torch.tensor([processed["attention_mask"]]),
        )
        scores = self.softmax(logits[0]).tolist()[0]
        predictions = []
        for score, label in zip(scores, self.lables):
            predictions.append({"label": label, "score": score})
        return predictions


if __name__ == "__main__":
    sentence = "The boy is sitting on a bench"
    predictor = ColaPredictor("./models/best-checkpoint.ckpt")
    print(predictor.predict(sentence))



================================================
FILE: week_2_hydra_config/model.py
================================================
import torch
import wandb
import hydra
import numpy as np
import pandas as pd
import torchmetrics
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification
from omegaconf import OmegaConf, DictConfig
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class ColaModel(pl.LightningModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", lr=3e-5):
        super(ColaModel, self).__init__()
        self.save_hyperparameters()

        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        self.num_classes = 2
        self.train_accuracy_metric = torchmetrics.Accuracy()
        self.val_accuracy_metric = torchmetrics.Accuracy()
        self.f1_metric = torchmetrics.F1(num_classes=self.num_classes)
        self.precision_macro_metric = torchmetrics.Precision(
            average="macro", num_classes=self.num_classes
        )
        self.recall_macro_metric = torchmetrics.Recall(
            average="macro", num_classes=self.num_classes
        )
        self.precision_micro_metric = torchmetrics.Precision(average="micro")
        self.recall_micro_metric = torchmetrics.Recall(average="micro")

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        # loss = F.cross_entropy(logits, batch["label"])
        preds = torch.argmax(outputs.logits, 1)
        train_acc = self.train_accuracy_metric(preds, batch["label"])
        self.log("train/loss", outputs.loss, prog_bar=True, on_epoch=True)
        self.log("train/acc", train_acc, prog_bar=True, on_epoch=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        labels = batch["label"]
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        preds = torch.argmax(outputs.logits, 1)

        # Metrics
        valid_acc = self.val_accuracy_metric(preds, labels)
        precision_macro = self.precision_macro_metric(preds, labels)
        recall_macro = self.recall_macro_metric(preds, labels)
        precision_micro = self.precision_micro_metric(preds, labels)
        recall_micro = self.recall_micro_metric(preds, labels)
        f1 = self.f1_metric(preds, labels)

        # Logging metrics
        self.log("valid/loss", outputs.loss, prog_bar=True, on_step=True)
        self.log("valid/acc", valid_acc, prog_bar=True, on_epoch=True)
        self.log("valid/precision_macro", precision_macro, prog_bar=True, on_epoch=True)
        self.log("valid/recall_macro", recall_macro, prog_bar=True, on_epoch=True)
        self.log("valid/precision_micro", precision_micro, prog_bar=True, on_epoch=True)
        self.log("valid/recall_micro", recall_micro, prog_bar=True, on_epoch=True)
        self.log("valid/f1", f1, prog_bar=True, on_epoch=True)
        return {"labels": labels, "logits": outputs.logits}

    def validation_epoch_end(self, outputs):
        labels = torch.cat([x["labels"] for x in outputs])
        logits = torch.cat([x["logits"] for x in outputs])
        preds = torch.argmax(logits, 1)

        ## There are multiple ways to track the metrics
        # 1. Confusion matrix plotting using inbuilt W&B method
        self.logger.experiment.log(
            {
                "conf": wandb.plot.confusion_matrix(
                    probs=logits.numpy(), y_true=labels.numpy()
                )
            }
        )

        # 2. Confusion Matrix plotting using scikit-learn method
        # wandb.log({"cm": wandb.sklearn.plot_confusion_matrix(labels.numpy(), preds)})

        # 3. Confusion Matric plotting using Seaborn
        # data = confusion_matrix(labels.numpy(), preds.numpy())
        # df_cm = pd.DataFrame(data, columns=np.unique(labels), index=np.unique(labels))
        # df_cm.index.name = "Actual"
        # df_cm.columns.name = "Predicted"
        # plt.figure(figsize=(7, 4))
        # plot = sns.heatmap(
        #     df_cm, cmap="Blues", annot=True, annot_kws={"size": 16}
        # )  # font size
        # self.logger.experiment.log({"Confusion Matrix": wandb.Image(plot)})

        # self.logger.experiment.log(
        #     {"roc": wandb.plot.roc_curve(labels.numpy(), logits.numpy())}
        # )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])



================================================
FILE: week_2_hydra_config/requirements.txt
================================================
pytorch-lightning==1.2.10
datasets==1.6.2
transformers==4.5.1
scikit-learn==0.24.2
wandb
torchmetrics
matplotlib
seaborn
hydra-core
omegaconf
hydra_colorlog


================================================
FILE: week_2_hydra_config/train.py
================================================
import torch
import hydra
import wandb
import logging

import pandas as pd
import pytorch_lightning as pl
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from data import DataModule
from model import ColaModel

logger = logging.getLogger(__name__)


class SamplesVisualisationLogger(pl.Callback):
    def __init__(self, datamodule):
        super().__init__()

        self.datamodule = datamodule

    def on_validation_end(self, trainer, pl_module):
        val_batch = next(iter(self.datamodule.val_dataloader()))
        sentences = val_batch["sentence"]

        outputs = pl_module(val_batch["input_ids"], val_batch["attention_mask"])
        preds = torch.argmax(outputs.logits, 1)
        labels = val_batch["label"]

        df = pd.DataFrame(
            {"Sentence": sentences, "Label": labels.numpy(), "Predicted": preds.numpy()}
        )

        wrong_df = df[df["Label"] != df["Predicted"]]
        trainer.logger.experiment.log(
            {
                "examples": wandb.Table(dataframe=wrong_df, allow_mixed_types=True),
                "global_step": trainer.global_step,
            }
        )


@hydra.main(config_path="./configs", config_name="config")
def main(cfg):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    logger.info(f"Using the model: {cfg.model.name}")
    logger.info(f"Using the tokenizer: {cfg.model.tokenizer}")
    cola_data = DataModule(
        cfg.model.tokenizer, cfg.processing.batch_size, cfg.processing.max_length
    )
    cola_model = ColaModel(cfg.model.name)

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models",
        filename="best-checkpoint",
        monitor="valid/loss",
        mode="min",
    )

    early_stopping_callback = EarlyStopping(
        monitor="valid/loss", patience=3, verbose=True, mode="min"
    )

    wandb_logger = WandbLogger(project="MLOps Basics", entity="raviraja")
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, SamplesVisualisationLogger(cola_data), early_stopping_callback],
        log_every_n_steps=cfg.training.log_every_n_steps,
        deterministic=cfg.training.deterministic,
        limit_train_batches=cfg.training.limit_train_batches,
        limit_val_batches=cfg.training.limit_val_batches,
    )
    trainer.fit(cola_model, cola_data)
    wandb.finish()


if __name__ == "__main__":
    main()



================================================
FILE: week_2_hydra_config/configs/config.yaml
================================================
defaults:
  - model: default
  - processing: default
  - training: default
  - override hydra/job_logging: colorlog
  - override hydra/hydra_logging: colorlog


================================================
FILE: week_2_hydra_config/configs/model/default.yaml
================================================
name: google/bert_uncased_L-2_H-128_A-2             # model used for training the classifier
tokenizer: google/bert_uncased_L-2_H-128_A-2        # tokenizer used for processing the data


================================================
FILE: week_2_hydra_config/configs/processing/default.yaml
================================================
batch_size: 64
max_length: 128


================================================
FILE: week_2_hydra_config/configs/training/default.yaml
================================================
max_epochs: 1
log_every_n_steps: 10
deterministic: true
limit_train_batches: 0.25
limit_val_batches: ${training.limit_train_batches}


================================================
FILE: week_2_hydra_config/experimental_notebooks/data_exploration.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
## Imports
"""

import datasets
import pandas as pd

from datasets import load_dataset

"""
## Dataset
"""

cola_dataset = load_dataset('glue', 'cola')
# Output:
#   Reusing dataset glue (/Users/raviraja/.cache/huggingface/datasets/glue/cola/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad)


cola_dataset
# Output:
#   DatasetDict({

#       train: Dataset({

#           features: ['sentence', 'label', 'idx'],

#           num_rows: 8551

#       })

#       validation: Dataset({

#           features: ['sentence', 'label', 'idx'],

#           num_rows: 1043

#       })

#       test: Dataset({

#           features: ['sentence', 'label', 'idx'],

#           num_rows: 1063

#       })

#   })

train_dataset = cola_dataset['train']
val_dataset = cola_dataset['validation']
test_dataset = cola_dataset['test']

len(train_dataset), len(val_dataset), len(test_dataset)
# Output:
#   (8551, 1043, 1063)

train_dataset[0]
# Output:
#   {'idx': 0,

#    'label': 1,

#    'sentence': "Our friends won't buy this analysis, let alone the next one we propose."}

val_dataset[0]
# Output:
#   {'idx': 0,

#    'label': 1,

#    'sentence': 'The sailors rode the breeze clear of the rocks.'}

test_dataset[0]
# Output:
#   {'idx': 0, 'label': -1, 'sentence': 'Bill whistled past the house.'}

train_dataset.features
# Output:
#   {'sentence': Value(dtype='string', id=None),

#    'label': ClassLabel(num_classes=2, names=['unacceptable', 'acceptable'], names_file=None, id=None),

#    'idx': Value(dtype='int32', id=None)}

train_dataset.filter(lambda example: example['label'] == train_dataset.features['label'].str2int('acceptable'))[:5]
# Output:
#   HBox(children=(FloatProgress(value=0.0, max=9.0), HTML(value='')))
#   

#   {'idx': [0, 1, 2, 3, 4],

#    'label': [1, 1, 1, 1, 1],

#    'sentence': ["Our friends won't buy this analysis, let alone the next one we propose.",

#     "One more pseudo generalization and I'm giving up.",

#     "One more pseudo generalization or I'm giving up.",

#     'The more we study verbs, the crazier they get.',

#     'Day by day the facts are getting murkier.']}

train_dataset.filter(lambda example: example['label'] == train_dataset.features['label'].str2int('unacceptable'))[:5]
# Output:
#   HBox(children=(FloatProgress(value=0.0, max=9.0), HTML(value='')))
#   

#   {'idx': [18, 20, 22, 23, 25],

#    'label': [0, 0, 0, 0, 0],

#    'sentence': ['They drank the pub.',

#     'The professor talked us.',

#     'We yelled ourselves.',

#     'We yelled Harry hoarse.',

#     'Harry coughed himself.']}

"""
## Tokenizing
"""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")

train_dataset = cola_dataset['train']
val_dataset = cola_dataset['validation']
test_dataset = cola_dataset['test']

tokenizer
# Output:
#   PreTrainedTokenizerFast(name_or_path='google/bert_uncased_L-2_H-128_A-2', vocab_size=30522, model_max_len=1000000000000000019884624838656, is_fast=True, padding_side='right', special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'})

print(train_dataset[0]['sentence'])
tokenizer(train_dataset[0]['sentence'])
# Output:
#   Our friends won't buy this analysis, let alone the next one we propose.

#   {'input_ids': [101, 2256, 2814, 2180, 1005, 1056, 4965, 2023, 4106, 1010, 2292, 2894, 1996, 2279, 2028, 2057, 16599, 1012, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

tokenizer.decode(tokenizer(train_dataset[0]['sentence'])['input_ids'])
# Output:
#   "[CLS] our friends won't buy this analysis, let alone the next one we propose. [SEP]"

def encode(examples):
    return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

train_dataset = train_dataset.map(encode, batched=True)
# Output:
#   HBox(children=(FloatProgress(value=0.0, max=9.0), HTML(value='')))
#   


"""
## Formatting
"""

import torch

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

"""
## Data Loader
"""

dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

next(iter(dataloader))
# Output:
#   {'attention_mask': tensor([[1, 1, 1,  ..., 0, 0, 0],

#            [1, 1, 1,  ..., 0, 0, 0],

#            [1, 1, 1,  ..., 0, 0, 0],

#            ...,

#            [1, 1, 1,  ..., 0, 0, 0],

#            [1, 1, 1,  ..., 0, 0, 0],

#            [1, 1, 1,  ..., 0, 0, 0]]),

#    'input_ids': tensor([[  101,  2256,  2814,  ...,     0,     0,     0],

#            [  101,  2028,  2062,  ...,     0,     0,     0],

#            [  101,  2028,  2062,  ...,     0,     0,     0],

#            ...,

#            [  101,  5965, 12808,  ...,     0,     0,     0],

#            [  101,  2198, 10948,  ...,     0,     0,     0],

#            [  101,  3021, 24471,  ...,     0,     0,     0]]),

#    'label': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0,

#            1, 0, 0, 1, 1, 1, 1, 1])}

for batch in dataloader:
    print(batch['input_ids'].shape, batch['attention_mask'].shape, batch['label'].shape)
# Output:
#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([7, 512]) torch.Size([7, 512]) torch.Size([7])




================================================
FILE: week_3_dvc/README.md
================================================

**Note: The purpose of the project to explore the libraries and learn how to use them. Not to build a SOTA model.**

## Requirements:

This project uses Python 3.8

Create a virtual env with the following command:

```
conda create --name project-setup python=3.8
conda activate project-setup
```

Install the requirements:

```
pip install -r requirements.txt
```

## Running

### Training

After installing the requirements, in order to train the model simply run:

```
python train.py
```

### Monitoring

Once the training is completed in the end of the logs you will see something like:

```
wandb: Synced 5 W&B file(s), 4 media file(s), 3 artifact file(s) and 0 other file(s)
wandb:
wandb: Synced proud-mountain-77: https://wandb.ai/raviraja/MLOps%20Basics/runs/3vp1twdc
```

Follow the link to see the wandb dashboard which contains all the plots.

### Inference

After training, update the model checkpoint path in the code and run

```
python inference.py
```

### Versioning data

Refer to the blog: [DVC Configuration](https://www.ravirajag.dev/blog/mlops-dvc)

### Running notebooks

I am using [Jupyter lab](https://jupyter.org/install) to run the notebooks.

Since I am using a virtualenv, when I run the command `jupyter lab` it might or might not use the virtualenv.

To make sure to use the virutalenv, run the following commands before running `jupyter lab`

```
conda install ipykernel
python -m ipykernel install --user --name project-setup
pip install ipywidgets
```


================================================
FILE: week_3_dvc/data.py
================================================
import torch
import pytorch_lightning as pl

from datasets import load_dataset
from transformers import AutoTokenizer


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name="google/bert_uncased_L-2_H-128_A-2",
        batch_size=64,
        max_length=128,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self):
        cola_dataset = load_dataset("glue", "cola")
        self.train_data = cola_dataset["train"]
        self.val_data = cola_dataset["validation"]

    def tokenize_data(self, example):
        return self.tokenizer(
            example["sentence"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

    def setup(self, stage=None):
        # we set up only relevant datasets when stage is specified
        if stage == "fit" or stage is None:
            self.train_data = self.train_data.map(self.tokenize_data, batched=True)
            self.train_data.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"]
            )

            self.val_data = self.val_data.map(self.tokenize_data, batched=True)
            self.val_data.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "label"],
                output_all_columns=True,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False
        )


if __name__ == "__main__":
    data_model = DataModule()
    data_model.prepare_data()
    data_model.setup()
    print(next(iter(data_model.train_dataloader()))["input_ids"].shape)



================================================
FILE: week_3_dvc/inference.py
================================================
import torch
from model import ColaModel
from data import DataModule


class ColaPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = ColaModel.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.freeze()
        self.processor = DataModule()
        self.softmax = torch.nn.Softmax(dim=0)
        self.lables = ["unacceptable", "acceptable"]

    def predict(self, text):
        inference_sample = {"sentence": text}
        processed = self.processor.tokenize_data(inference_sample)
        logits = self.model(
            torch.tensor([processed["input_ids"]]),
            torch.tensor([processed["attention_mask"]]),
        )
        scores = self.softmax(logits[0]).tolist()[0]
        predictions = []
        for score, label in zip(scores, self.lables):
            predictions.append({"label": label, "score": score})
        return predictions


if __name__ == "__main__":
    sentence = "The boy is sitting on a bench"
    predictor = ColaPredictor("./models/best-checkpoint.ckpt")
    print(predictor.predict(sentence))



================================================
FILE: week_3_dvc/model.py
================================================
import torch
import wandb
import hydra
import numpy as np
import pandas as pd
import torchmetrics
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification
from omegaconf import OmegaConf, DictConfig
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class ColaModel(pl.LightningModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", lr=3e-5):
        super(ColaModel, self).__init__()
        self.save_hyperparameters()

        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        self.num_classes = 2
        self.train_accuracy_metric = torchmetrics.Accuracy()
        self.val_accuracy_metric = torchmetrics.Accuracy()
        self.f1_metric = torchmetrics.F1(num_classes=self.num_classes)
        self.precision_macro_metric = torchmetrics.Precision(
            average="macro", num_classes=self.num_classes
        )
        self.recall_macro_metric = torchmetrics.Recall(
            average="macro", num_classes=self.num_classes
        )
        self.precision_micro_metric = torchmetrics.Precision(average="micro")
        self.recall_micro_metric = torchmetrics.Recall(average="micro")

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        # loss = F.cross_entropy(logits, batch["label"])
        preds = torch.argmax(outputs.logits, 1)
        train_acc = self.train_accuracy_metric(preds, batch["label"])
        self.log("train/loss", outputs.loss, prog_bar=True, on_epoch=True)
        self.log("train/acc", train_acc, prog_bar=True, on_epoch=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        labels = batch["label"]
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        preds = torch.argmax(outputs.logits, 1)

        # Metrics
        valid_acc = self.val_accuracy_metric(preds, labels)
        precision_macro = self.precision_macro_metric(preds, labels)
        recall_macro = self.recall_macro_metric(preds, labels)
        precision_micro = self.precision_micro_metric(preds, labels)
        recall_micro = self.recall_micro_metric(preds, labels)
        f1 = self.f1_metric(preds, labels)

        # Logging metrics
        self.log("valid/loss", outputs.loss, prog_bar=True, on_step=True)
        self.log("valid/acc", valid_acc, prog_bar=True, on_epoch=True)
        self.log("valid/precision_macro", precision_macro, prog_bar=True, on_epoch=True)
        self.log("valid/recall_macro", recall_macro, prog_bar=True, on_epoch=True)
        self.log("valid/precision_micro", precision_micro, prog_bar=True, on_epoch=True)
        self.log("valid/recall_micro", recall_micro, prog_bar=True, on_epoch=True)
        self.log("valid/f1", f1, prog_bar=True, on_epoch=True)
        return {"labels": labels, "logits": outputs.logits}

    def validation_epoch_end(self, outputs):
        labels = torch.cat([x["labels"] for x in outputs])
        logits = torch.cat([x["logits"] for x in outputs])
        preds = torch.argmax(logits, 1)

        ## There are multiple ways to track the metrics
        # 1. Confusion matrix plotting using inbuilt W&B method
        self.logger.experiment.log(
            {
                "conf": wandb.plot.confusion_matrix(
                    probs=logits.numpy(), y_true=labels.numpy()
                )
            }
        )

        # 2. Confusion Matrix plotting using scikit-learn method
        # wandb.log({"cm": wandb.sklearn.plot_confusion_matrix(labels.numpy(), preds)})

        # 3. Confusion Matric plotting using Seaborn
        # data = confusion_matrix(labels.numpy(), preds.numpy())
        # df_cm = pd.DataFrame(data, columns=np.unique(labels), index=np.unique(labels))
        # df_cm.index.name = "Actual"
        # df_cm.columns.name = "Predicted"
        # plt.figure(figsize=(7, 4))
        # plot = sns.heatmap(
        #     df_cm, cmap="Blues", annot=True, annot_kws={"size": 16}
        # )  # font size
        # self.logger.experiment.log({"Confusion Matrix": wandb.Image(plot)})

        # self.logger.experiment.log(
        #     {"roc": wandb.plot.roc_curve(labels.numpy(), logits.numpy())}
        # )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])



================================================
FILE: week_3_dvc/requirements.txt
================================================
pytorch-lightning==1.2.10
datasets==1.6.2
transformers==4.5.1
scikit-learn==0.24.2
wandb
torchmetrics
matplotlib
seaborn
hydra-core
omegaconf
hydra_colorlog


================================================
FILE: week_3_dvc/train.py
================================================
import torch
import hydra
import wandb
import logging

import pandas as pd
import pytorch_lightning as pl
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from data import DataModule
from model import ColaModel

logger = logging.getLogger(__name__)


class SamplesVisualisationLogger(pl.Callback):
    def __init__(self, datamodule):
        super().__init__()

        self.datamodule = datamodule

    def on_validation_end(self, trainer, pl_module):
        val_batch = next(iter(self.datamodule.val_dataloader()))
        sentences = val_batch["sentence"]

        outputs = pl_module(val_batch["input_ids"], val_batch["attention_mask"])
        preds = torch.argmax(outputs.logits, 1)
        labels = val_batch["label"]

        df = pd.DataFrame(
            {"Sentence": sentences, "Label": labels.numpy(), "Predicted": preds.numpy()}
        )

        wrong_df = df[df["Label"] != df["Predicted"]]
        trainer.logger.experiment.log(
            {
                "examples": wandb.Table(dataframe=wrong_df, allow_mixed_types=True),
                "global_step": trainer.global_step,
            }
        )


@hydra.main(config_path="./configs", config_name="config")
def main(cfg):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    logger.info(f"Using the model: {cfg.model.name}")
    logger.info(f"Using the tokenizer: {cfg.model.tokenizer}")
    cola_data = DataModule(
        cfg.model.tokenizer, cfg.processing.batch_size, cfg.processing.max_length
    )
    cola_model = ColaModel(cfg.model.name)

    root_dir = hydra.utils.get_original_cwd()
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{root_dir}/models",
        filename="best-checkpoint",
        monitor="valid/loss",
        mode="min",
    )

    early_stopping_callback = EarlyStopping(
        monitor="valid/loss", patience=3, verbose=True, mode="min"
    )

    wandb_logger = WandbLogger(project="MLOps Basics", entity="raviraja")
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, SamplesVisualisationLogger(cola_data), early_stopping_callback],
        log_every_n_steps=cfg.training.log_every_n_steps,
        deterministic=cfg.training.deterministic,
        # limit_train_batches=cfg.training.limit_train_batches,
        # limit_val_batches=cfg.training.limit_val_batches,
    )
    trainer.fit(cola_model, cola_data)
    wandb.finish()


if __name__ == "__main__":
    main()



================================================
FILE: week_3_dvc/configs/config.yaml
================================================
defaults:
  - model: default
  - processing: default
  - training: default
  - override hydra/job_logging: colorlog
  - override hydra/hydra_logging: colorlog


================================================
FILE: week_3_dvc/configs/model/default.yaml
================================================
name: google/bert_uncased_L-2_H-128_A-2             # model used for training the classifier
tokenizer: google/bert_uncased_L-2_H-128_A-2        # tokenizer used for processing the data


================================================
FILE: week_3_dvc/configs/processing/default.yaml
================================================
batch_size: 64
max_length: 128


================================================
FILE: week_3_dvc/configs/training/default.yaml
================================================
max_epochs: 1
log_every_n_steps: 10
deterministic: true
limit_train_batches: 0.25
limit_val_batches: ${training.limit_train_batches}


================================================
FILE: week_3_dvc/dvcfiles/trained_model.dvc
================================================
wdir: ../models
outs:
- md5: c2f5c0a1954209865b9be1945f33ed6e
  size: 17567709
  path: best-checkpoint.ckpt



================================================
FILE: week_3_dvc/experimental_notebooks/data_exploration.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
## Imports
"""

import datasets
import pandas as pd

from datasets import load_dataset

"""
## Dataset
"""

cola_dataset = load_dataset('glue', 'cola')
# Output:
#   Reusing dataset glue (/Users/raviraja/.cache/huggingface/datasets/glue/cola/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad)


cola_dataset
# Output:
#   DatasetDict({

#       train: Dataset({

#           features: ['sentence', 'label', 'idx'],

#           num_rows: 8551

#       })

#       validation: Dataset({

#           features: ['sentence', 'label', 'idx'],

#           num_rows: 1043

#       })

#       test: Dataset({

#           features: ['sentence', 'label', 'idx'],

#           num_rows: 1063

#       })

#   })

train_dataset = cola_dataset['train']
val_dataset = cola_dataset['validation']
test_dataset = cola_dataset['test']

len(train_dataset), len(val_dataset), len(test_dataset)
# Output:
#   (8551, 1043, 1063)

train_dataset[0]
# Output:
#   {'idx': 0,

#    'label': 1,

#    'sentence': "Our friends won't buy this analysis, let alone the next one we propose."}

val_dataset[0]
# Output:
#   {'idx': 0,

#    'label': 1,

#    'sentence': 'The sailors rode the breeze clear of the rocks.'}

test_dataset[0]
# Output:
#   {'idx': 0, 'label': -1, 'sentence': 'Bill whistled past the house.'}

train_dataset.features
# Output:
#   {'sentence': Value(dtype='string', id=None),

#    'label': ClassLabel(num_classes=2, names=['unacceptable', 'acceptable'], names_file=None, id=None),

#    'idx': Value(dtype='int32', id=None)}

train_dataset.filter(lambda example: example['label'] == train_dataset.features['label'].str2int('acceptable'))[:5]
# Output:
#   HBox(children=(FloatProgress(value=0.0, max=9.0), HTML(value='')))
#   

#   {'idx': [0, 1, 2, 3, 4],

#    'label': [1, 1, 1, 1, 1],

#    'sentence': ["Our friends won't buy this analysis, let alone the next one we propose.",

#     "One more pseudo generalization and I'm giving up.",

#     "One more pseudo generalization or I'm giving up.",

#     'The more we study verbs, the crazier they get.',

#     'Day by day the facts are getting murkier.']}

train_dataset.filter(lambda example: example['label'] == train_dataset.features['label'].str2int('unacceptable'))[:5]
# Output:
#   HBox(children=(FloatProgress(value=0.0, max=9.0), HTML(value='')))
#   

#   {'idx': [18, 20, 22, 23, 25],

#    'label': [0, 0, 0, 0, 0],

#    'sentence': ['They drank the pub.',

#     'The professor talked us.',

#     'We yelled ourselves.',

#     'We yelled Harry hoarse.',

#     'Harry coughed himself.']}

"""
## Tokenizing
"""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")

train_dataset = cola_dataset['train']
val_dataset = cola_dataset['validation']
test_dataset = cola_dataset['test']

tokenizer
# Output:
#   PreTrainedTokenizerFast(name_or_path='google/bert_uncased_L-2_H-128_A-2', vocab_size=30522, model_max_len=1000000000000000019884624838656, is_fast=True, padding_side='right', special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'})

print(train_dataset[0]['sentence'])
tokenizer(train_dataset[0]['sentence'])
# Output:
#   Our friends won't buy this analysis, let alone the next one we propose.

#   {'input_ids': [101, 2256, 2814, 2180, 1005, 1056, 4965, 2023, 4106, 1010, 2292, 2894, 1996, 2279, 2028, 2057, 16599, 1012, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

tokenizer.decode(tokenizer(train_dataset[0]['sentence'])['input_ids'])
# Output:
#   "[CLS] our friends won't buy this analysis, let alone the next one we propose. [SEP]"

def encode(examples):
    return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

train_dataset = train_dataset.map(encode, batched=True)
# Output:
#   HBox(children=(FloatProgress(value=0.0, max=9.0), HTML(value='')))
#   


"""
## Formatting
"""

import torch

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

"""
## Data Loader
"""

dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

next(iter(dataloader))
# Output:
#   {'attention_mask': tensor([[1, 1, 1,  ..., 0, 0, 0],

#            [1, 1, 1,  ..., 0, 0, 0],

#            [1, 1, 1,  ..., 0, 0, 0],

#            ...,

#            [1, 1, 1,  ..., 0, 0, 0],

#            [1, 1, 1,  ..., 0, 0, 0],

#            [1, 1, 1,  ..., 0, 0, 0]]),

#    'input_ids': tensor([[  101,  2256,  2814,  ...,     0,     0,     0],

#            [  101,  2028,  2062,  ...,     0,     0,     0],

#            [  101,  2028,  2062,  ...,     0,     0,     0],

#            ...,

#            [  101,  5965, 12808,  ...,     0,     0,     0],

#            [  101,  2198, 10948,  ...,     0,     0,     0],

#            [  101,  3021, 24471,  ...,     0,     0,     0]]),

#    'label': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0,

#            1, 0, 0, 1, 1, 1, 1, 1])}

for batch in dataloader:
    print(batch['input_ids'].shape, batch['attention_mask'].shape, batch['label'].shape)
# Output:
#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([7, 512]) torch.Size([7, 512]) torch.Size([7])




================================================
FILE: week_3_dvc/models/.gitignore
================================================
/best-checkpoint.ckpt



================================================
FILE: week_4_onnx/README.md
================================================

**Note: The purpose of the project to explore the libraries and learn how to use them. Not to build a SOTA model.**

## Requirements:

This project uses Python 3.8

Create a virtual env with the following command:

```
conda create --name project-setup python=3.8
conda activate project-setup
```

Install the requirements:

```
pip install -r requirements.txt
```

## Running

### Training

After installing the requirements, in order to train the model simply run:

```
python train.py
```

### Monitoring

Once the training is completed in the end of the logs you will see something like:

```
wandb: Synced 5 W&B file(s), 4 media file(s), 3 artifact file(s) and 0 other file(s)
wandb:
wandb: Synced proud-mountain-77: https://wandb.ai/raviraja/MLOps%20Basics/runs/3vp1twdc
```

Follow the link to see the wandb dashboard which contains all the plots.

### Versioning data

Refer to the blog: [DVC Configuration](https://www.ravirajag.dev/blog/mlops-dvc)

### Exporting model to ONNX

Once the model is trained, convert the model using the following command:

```
python convert_model_to_onnx.py
```

### Inference

#### Inference using standard pytorch

```
python inference.py
```

#### Inference using ONNX Runtime

```
python inference_onnx.py
```


### Running notebooks

I am using [Jupyter lab](https://jupyter.org/install) to run the notebooks.

Since I am using a virtualenv, when I run the command `jupyter lab` it might or might not use the virtualenv.

To make sure to use the virutalenv, run the following commands before running `jupyter lab`

```
conda install ipykernel
python -m ipykernel install --user --name project-setup
pip install ipywidgets
```


================================================
FILE: week_4_onnx/convert_model_to_onnx.py
================================================
import torch
import hydra
import logging

from omegaconf.omegaconf import OmegaConf

from model import ColaModel
from data import DataModule

logger = logging.getLogger(__name__)


@hydra.main(config_path="./configs", config_name="config")
def convert_model(cfg):
    root_dir = hydra.utils.get_original_cwd()
    model_path = f"{root_dir}/models/best-checkpoint.ckpt"
    logger.info(f"Loading pre-trained model from: {model_path}")
    cola_model = ColaModel.load_from_checkpoint(model_path)

    data_model = DataModule(
        cfg.model.tokenizer, cfg.processing.batch_size, cfg.processing.max_length
    )
    data_model.prepare_data()
    data_model.setup()
    input_batch = next(iter(data_model.train_dataloader()))
    input_sample = {
        "input_ids": input_batch["input_ids"][0].unsqueeze(0),
        "attention_mask": input_batch["attention_mask"][0].unsqueeze(0),
    }

    # Export the model
    logger.info(f"Converting the model into ONNX format")
    torch.onnx.export(
        cola_model,  # model being run
        (
            input_sample["input_ids"],
            input_sample["attention_mask"],
        ),  # model input (or a tuple for multiple inputs)
        f"{root_dir}/models/model.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,
        opset_version=10,
        input_names=["input_ids", "attention_mask"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input_ids": {0: "batch_size"},  # variable length axes
            "attention_mask": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    logger.info(
        f"Model converted successfully. ONNX format model is at: {root_dir}/models/model.onnx"
    )


if __name__ == "__main__":
    convert_model()



================================================
FILE: week_4_onnx/data.py
================================================
import torch
import datasets
import pytorch_lightning as pl

from datasets import load_dataset
from transformers import AutoTokenizer


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name="google/bert_uncased_L-2_H-128_A-2",
        batch_size=64,
        max_length=128,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self):
        cola_dataset = load_dataset("glue", "cola")
        self.train_data = cola_dataset["train"]
        self.val_data = cola_dataset["validation"]

    def tokenize_data(self, example):
        return self.tokenizer(
            example["sentence"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

    def setup(self, stage=None):
        # we set up only relevant datasets when stage is specified
        if stage == "fit" or stage is None:
            self.train_data = self.train_data.map(self.tokenize_data, batched=True)
            self.train_data.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"]
            )

            self.val_data = self.val_data.map(self.tokenize_data, batched=True)
            self.val_data.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "label"],
                output_all_columns=True,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False
        )


if __name__ == "__main__":
    data_model = DataModule()
    data_model.prepare_data()
    data_model.setup()
    print(next(iter(data_model.train_dataloader()))["input_ids"].shape)



================================================
FILE: week_4_onnx/inference.py
================================================
import torch
from model import ColaModel
from data import DataModule
from utils import timing


class ColaPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = ColaModel.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.freeze()
        self.processor = DataModule()
        self.softmax = torch.nn.Softmax(dim=1)
        self.lables = ["unacceptable", "acceptable"]

    @timing
    def predict(self, text):
        inference_sample = {"sentence": text}
        processed = self.processor.tokenize_data(inference_sample)
        logits = self.model(
            torch.tensor([processed["input_ids"]]),
            torch.tensor([processed["attention_mask"]]),
        )
        scores = self.softmax(logits[0]).tolist()[0]
        predictions = []
        for score, label in zip(scores, self.lables):
            predictions.append({"label": label, "score": score})
        return predictions


if __name__ == "__main__":
    sentence = "The boy is sitting on a bench"
    predictor = ColaPredictor("./models/best-checkpoint.ckpt")
    print(predictor.predict(sentence))
    sentences = ["The boy is sitting on a bench"] * 10
    for sentence in sentences:
        predictor.predict(sentence)



================================================
FILE: week_4_onnx/inference_onnx.py
================================================
import numpy as np
import onnxruntime as ort
from scipy.special import softmax

from data import DataModule
from utils import timing


class ColaONNXPredictor:
    def __init__(self, model_path):
        self.ort_session = ort.InferenceSession(model_path)
        self.processor = DataModule()
        self.lables = ["unacceptable", "acceptable"]

    @timing
    def predict(self, text):
        inference_sample = {"sentence": text}
        processed = self.processor.tokenize_data(inference_sample)

        ort_inputs = {
            "input_ids": np.expand_dims(processed["input_ids"], axis=0),
            "attention_mask": np.expand_dims(processed["attention_mask"], axis=0),
        }
        ort_outs = self.ort_session.run(None, ort_inputs)
        scores = softmax(ort_outs[0])[0]
        predictions = []
        for score, label in zip(scores, self.lables):
            predictions.append({"label": label, "score": score})
        return predictions


if __name__ == "__main__":
    sentence = "The boy is sitting on a bench"
    predictor = ColaONNXPredictor("./models/model.onnx")
    print(predictor.predict(sentence))
    sentences = ["The boy is sitting on a bench"] * 10
    for sentence in sentences:
        predictor.predict(sentence)



================================================
FILE: week_4_onnx/model.py
================================================
import torch
import wandb
import hydra
import numpy as np
import pandas as pd
import torchmetrics
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification
from omegaconf import OmegaConf, DictConfig
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class ColaModel(pl.LightningModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", lr=3e-5):
        super(ColaModel, self).__init__()
        self.save_hyperparameters()

        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        self.num_classes = 2
        self.train_accuracy_metric = torchmetrics.Accuracy()
        self.val_accuracy_metric = torchmetrics.Accuracy()
        self.f1_metric = torchmetrics.F1(num_classes=self.num_classes)
        self.precision_macro_metric = torchmetrics.Precision(
            average="macro", num_classes=self.num_classes
        )
        self.recall_macro_metric = torchmetrics.Recall(
            average="macro", num_classes=self.num_classes
        )
        self.precision_micro_metric = torchmetrics.Precision(average="micro")
        self.recall_micro_metric = torchmetrics.Recall(average="micro")

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        # loss = F.cross_entropy(logits, batch["label"])
        preds = torch.argmax(outputs.logits, 1)
        train_acc = self.train_accuracy_metric(preds, batch["label"])
        self.log("train/loss", outputs.loss, prog_bar=True, on_epoch=True)
        self.log("train/acc", train_acc, prog_bar=True, on_epoch=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        labels = batch["label"]
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        preds = torch.argmax(outputs.logits, 1)

        # Metrics
        valid_acc = self.val_accuracy_metric(preds, labels)
        precision_macro = self.precision_macro_metric(preds, labels)
        recall_macro = self.recall_macro_metric(preds, labels)
        precision_micro = self.precision_micro_metric(preds, labels)
        recall_micro = self.recall_micro_metric(preds, labels)
        f1 = self.f1_metric(preds, labels)

        # Logging metrics
        self.log("valid/loss", outputs.loss, prog_bar=True, on_step=True)
        self.log("valid/acc", valid_acc, prog_bar=True, on_epoch=True)
        self.log("valid/precision_macro", precision_macro, prog_bar=True, on_epoch=True)
        self.log("valid/recall_macro", recall_macro, prog_bar=True, on_epoch=True)
        self.log("valid/precision_micro", precision_micro, prog_bar=True, on_epoch=True)
        self.log("valid/recall_micro", recall_micro, prog_bar=True, on_epoch=True)
        self.log("valid/f1", f1, prog_bar=True, on_epoch=True)
        return {"labels": labels, "logits": outputs.logits}

    def validation_epoch_end(self, outputs):
        labels = torch.cat([x["labels"] for x in outputs])
        logits = torch.cat([x["logits"] for x in outputs])
        preds = torch.argmax(logits, 1)

        ## There are multiple ways to track the metrics
        # 1. Confusion matrix plotting using inbuilt W&B method
        self.logger.experiment.log(
            {
                "conf": wandb.plot.confusion_matrix(
                    probs=logits.numpy(), y_true=labels.numpy()
                )
            }
        )

        # 2. Confusion Matrix plotting using scikit-learn method
        # wandb.log({"cm": wandb.sklearn.plot_confusion_matrix(labels.numpy(), preds)})

        # 3. Confusion Matric plotting using Seaborn
        # data = confusion_matrix(labels.numpy(), preds.numpy())
        # df_cm = pd.DataFrame(data, columns=np.unique(labels), index=np.unique(labels))
        # df_cm.index.name = "Actual"
        # df_cm.columns.name = "Predicted"
        # plt.figure(figsize=(7, 4))
        # plot = sns.heatmap(
        #     df_cm, cmap="Blues", annot=True, annot_kws={"size": 16}
        # )  # font size
        # self.logger.experiment.log({"Confusion Matrix": wandb.Image(plot)})

        # self.logger.experiment.log(
        #     {"roc": wandb.plot.roc_curve(labels.numpy(), logits.numpy())}
        # )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])



================================================
FILE: week_4_onnx/requirements.txt
================================================
pytorch-lightning==1.2.10
datasets==1.6.2
transformers==4.5.1
scikit-learn==0.24.2
wandb
torchmetrics
matplotlib
seaborn
hydra-core
omegaconf
hydra_colorlog


================================================
FILE: week_4_onnx/train.py
================================================
import torch
import hydra
import wandb
import logging

import pandas as pd
import pytorch_lightning as pl
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from data import DataModule
from model import ColaModel

logger = logging.getLogger(__name__)


class SamplesVisualisationLogger(pl.Callback):
    def __init__(self, datamodule):
        super().__init__()

        self.datamodule = datamodule

    def on_validation_end(self, trainer, pl_module):
        val_batch = next(iter(self.datamodule.val_dataloader()))
        sentences = val_batch["sentence"]

        outputs = pl_module(val_batch["input_ids"], val_batch["attention_mask"])
        preds = torch.argmax(outputs.logits, 1)
        labels = val_batch["label"]

        df = pd.DataFrame(
            {"Sentence": sentences, "Label": labels.numpy(), "Predicted": preds.numpy()}
        )

        wrong_df = df[df["Label"] != df["Predicted"]]
        trainer.logger.experiment.log(
            {
                "examples": wandb.Table(dataframe=wrong_df, allow_mixed_types=True),
                "global_step": trainer.global_step,
            }
        )


@hydra.main(config_path="./configs", config_name="config")
def main(cfg):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    logger.info(f"Using the model: {cfg.model.name}")
    logger.info(f"Using the tokenizer: {cfg.model.tokenizer}")
    cola_data = DataModule(
        cfg.model.tokenizer, cfg.processing.batch_size, cfg.processing.max_length
    )
    cola_model = ColaModel(cfg.model.name)

    root_dir = hydra.utils.get_original_cwd()
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{root_dir}/models",
        filename="best-checkpoint",
        monitor="valid/loss",
        mode="min",
    )

    early_stopping_callback = EarlyStopping(
        monitor="valid/loss", patience=3, verbose=True, mode="min"
    )

    wandb_logger = WandbLogger(project="MLOps Basics", entity="raviraja")
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, SamplesVisualisationLogger(cola_data), early_stopping_callback],
        log_every_n_steps=cfg.training.log_every_n_steps,
        deterministic=cfg.training.deterministic,
        # limit_train_batches=cfg.training.limit_train_batches,
        # limit_val_batches=cfg.training.limit_val_batches,
    )
    trainer.fit(cola_model, cola_data)
    wandb.finish()


if __name__ == "__main__":
    main()



================================================
FILE: week_4_onnx/utils.py
================================================
import time
from functools import wraps


def timing(f):
    """Decorator for timing functions
    Usage:
    @timing
    def function(a):
        pass
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print("function:%r took: %2.5f sec" % (f.__name__, end - start))
        return result

    return wrapper



================================================
FILE: week_4_onnx/configs/config.yaml
================================================
defaults:
  - model: default
  - processing: default
  - training: default
  - override hydra/job_logging: colorlog
  - override hydra/hydra_logging: colorlog


================================================
FILE: week_4_onnx/configs/model/default.yaml
================================================
name: google/bert_uncased_L-2_H-128_A-2             # model used for training the classifier
tokenizer: google/bert_uncased_L-2_H-128_A-2        # tokenizer used for processing the data


================================================
FILE: week_4_onnx/configs/processing/default.yaml
================================================
batch_size: 64
max_length: 128


================================================
FILE: week_4_onnx/configs/training/default.yaml
================================================
max_epochs: 1
log_every_n_steps: 10
deterministic: true
limit_train_batches: 0.25
limit_val_batches: ${training.limit_train_batches}


================================================
FILE: week_4_onnx/dvcfiles/trained_model.dvc
================================================
wdir: ../models
outs:
- md5: c2f5c0a1954209865b9be1945f33ed6e
  size: 17567709
  path: best-checkpoint.ckpt



================================================
FILE: week_4_onnx/experimental_notebooks/data_exploration.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
## Imports
"""

import datasets
import pandas as pd

from datasets import load_dataset

"""
## Dataset
"""

cola_dataset = load_dataset('glue', 'cola')
# Output:
#   Reusing dataset glue (/Users/raviraja/.cache/huggingface/datasets/glue/cola/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad)


cola_dataset
# Output:
#   DatasetDict({

#       train: Dataset({

#           features: ['sentence', 'label', 'idx'],

#           num_rows: 8551

#       })

#       validation: Dataset({

#           features: ['sentence', 'label', 'idx'],

#           num_rows: 1043

#       })

#       test: Dataset({

#           features: ['sentence', 'label', 'idx'],

#           num_rows: 1063

#       })

#   })

train_dataset = cola_dataset['train']
val_dataset = cola_dataset['validation']
test_dataset = cola_dataset['test']

len(train_dataset), len(val_dataset), len(test_dataset)
# Output:
#   (8551, 1043, 1063)

train_dataset[0]
# Output:
#   {'idx': 0,

#    'label': 1,

#    'sentence': "Our friends won't buy this analysis, let alone the next one we propose."}

val_dataset[0]
# Output:
#   {'idx': 0,

#    'label': 1,

#    'sentence': 'The sailors rode the breeze clear of the rocks.'}

test_dataset[0]
# Output:
#   {'idx': 0, 'label': -1, 'sentence': 'Bill whistled past the house.'}

train_dataset.features
# Output:
#   {'sentence': Value(dtype='string', id=None),

#    'label': ClassLabel(num_classes=2, names=['unacceptable', 'acceptable'], names_file=None, id=None),

#    'idx': Value(dtype='int32', id=None)}

train_dataset.filter(lambda example: example['label'] == train_dataset.features['label'].str2int('acceptable'))[:5]
# Output:
#   HBox(children=(FloatProgress(value=0.0, max=9.0), HTML(value='')))
#   

#   {'idx': [0, 1, 2, 3, 4],

#    'label': [1, 1, 1, 1, 1],

#    'sentence': ["Our friends won't buy this analysis, let alone the next one we propose.",

#     "One more pseudo generalization and I'm giving up.",

#     "One more pseudo generalization or I'm giving up.",

#     'The more we study verbs, the crazier they get.',

#     'Day by day the facts are getting murkier.']}

train_dataset.filter(lambda example: example['label'] == train_dataset.features['label'].str2int('unacceptable'))[:5]
# Output:
#   HBox(children=(FloatProgress(value=0.0, max=9.0), HTML(value='')))
#   

#   {'idx': [18, 20, 22, 23, 25],

#    'label': [0, 0, 0, 0, 0],

#    'sentence': ['They drank the pub.',

#     'The professor talked us.',

#     'We yelled ourselves.',

#     'We yelled Harry hoarse.',

#     'Harry coughed himself.']}

"""
## Tokenizing
"""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")

train_dataset = cola_dataset['train']
val_dataset = cola_dataset['validation']
test_dataset = cola_dataset['test']

tokenizer
# Output:
#   PreTrainedTokenizerFast(name_or_path='google/bert_uncased_L-2_H-128_A-2', vocab_size=30522, model_max_len=1000000000000000019884624838656, is_fast=True, padding_side='right', special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'})

print(train_dataset[0]['sentence'])
tokenizer(train_dataset[0]['sentence'])
# Output:
#   Our friends won't buy this analysis, let alone the next one we propose.

#   {'input_ids': [101, 2256, 2814, 2180, 1005, 1056, 4965, 2023, 4106, 1010, 2292, 2894, 1996, 2279, 2028, 2057, 16599, 1012, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

tokenizer.decode(tokenizer(train_dataset[0]['sentence'])['input_ids'])
# Output:
#   "[CLS] our friends won't buy this analysis, let alone the next one we propose. [SEP]"

def encode(examples):
    return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

train_dataset = train_dataset.map(encode, batched=True)
# Output:
#   HBox(children=(FloatProgress(value=0.0, max=9.0), HTML(value='')))
#   


"""
## Formatting
"""

import torch

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

"""
## Data Loader
"""

dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

next(iter(dataloader))
# Output:
#   {'attention_mask': tensor([[1, 1, 1,  ..., 0, 0, 0],

#            [1, 1, 1,  ..., 0, 0, 0],

#            [1, 1, 1,  ..., 0, 0, 0],

#            ...,

#            [1, 1, 1,  ..., 0, 0, 0],

#            [1, 1, 1,  ..., 0, 0, 0],

#            [1, 1, 1,  ..., 0, 0, 0]]),

#    'input_ids': tensor([[  101,  2256,  2814,  ...,     0,     0,     0],

#            [  101,  2028,  2062,  ...,     0,     0,     0],

#            [  101,  2028,  2062,  ...,     0,     0,     0],

#            ...,

#            [  101,  5965, 12808,  ...,     0,     0,     0],

#            [  101,  2198, 10948,  ...,     0,     0,     0],

#            [  101,  3021, 24471,  ...,     0,     0,     0]]),

#    'label': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0,

#            1, 0, 0, 1, 1, 1, 1, 1])}

for batch in dataloader:
    print(batch['input_ids'].shape, batch['attention_mask'].shape, batch['label'].shape)
# Output:
#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([7, 512]) torch.Size([7, 512]) torch.Size([7])




================================================
FILE: week_5_docker/README.md
================================================

**Note: The purpose of the project to explore the libraries and learn how to use them. Not to build a SOTA model.**

## Requirements:

This project uses Python 3.8

Create a virtual env with the following command:

```
conda create --name project-setup python=3.8
conda activate project-setup
```

Install the requirements:

```
pip install -r requirements.txt
```

## Running

### Training

After installing the requirements, in order to train the model simply run:

```
python train.py
```

### Monitoring

Once the training is completed in the end of the logs you will see something like:

```
wandb: Synced 5 W&B file(s), 4 media file(s), 3 artifact file(s) and 0 other file(s)
wandb:
wandb: Synced proud-mountain-77: https://wandb.ai/raviraja/MLOps%20Basics/runs/3vp1twdc
```

Follow the link to see the wandb dashboard which contains all the plots.

### Versioning data

Refer to the blog: [DVC Configuration](https://www.ravirajag.dev/blog/mlops-dvc)

### Exporting model to ONNX

Once the model is trained, convert the model using the following command:

```
python convert_model_to_onnx.py
```

### Inference

#### Inference using standard pytorch

```
python inference.py
```

#### Inference using ONNX Runtime

```
python inference_onnx.py
```

### Docker

Install the docker using the [instructions here](https://docs.docker.com/engine/install/)

Build the image using the command

```shell
docker build -t inference:latest .
```

Then run the container using the command

```shell
docker run -p 8000:8000 --name inference_container inference:latest
```

(or)

Build and run the container using the command

```shell
docker-compose up
```


### Running notebooks

I am using [Jupyter lab](https://jupyter.org/install) to run the notebooks.

Since I am using a virtualenv, when I run the command `jupyter lab` it might or might not use the virtualenv.

To make sure to use the virutalenv, run the following commands before running `jupyter lab`

```
conda install ipykernel
python -m ipykernel install --user --name project-setup
pip install ipywidgets
```


================================================
FILE: week_5_docker/app.py
================================================
from fastapi import FastAPI
from inference_onnx import ColaONNXPredictor
app = FastAPI(title="MLOps Basics App")

predictor = ColaONNXPredictor("./models/model.onnx")

@app.get("/")
async def home_page():
    return "<h2>Sample prediction API</h2>"


@app.get("/predict")
async def get_prediction(text: str):
    result =  predictor.predict(text)
    return result


================================================
FILE: week_5_docker/convert_model_to_onnx.py
================================================
import torch
import hydra
import logging

from omegaconf.omegaconf import OmegaConf

from model import ColaModel
from data import DataModule

logger = logging.getLogger(__name__)


@hydra.main(config_path="./configs", config_name="config")
def convert_model(cfg):
    root_dir = hydra.utils.get_original_cwd()
    model_path = f"{root_dir}/models/best-checkpoint.ckpt"
    logger.info(f"Loading pre-trained model from: {model_path}")
    cola_model = ColaModel.load_from_checkpoint(model_path)

    data_model = DataModule(
        cfg.model.tokenizer, cfg.processing.batch_size, cfg.processing.max_length
    )
    data_model.prepare_data()
    data_model.setup()
    input_batch = next(iter(data_model.train_dataloader()))
    input_sample = {
        "input_ids": input_batch["input_ids"][0].unsqueeze(0),
        "attention_mask": input_batch["attention_mask"][0].unsqueeze(0),
    }

    # Export the model
    logger.info(f"Converting the model into ONNX format")
    torch.onnx.export(
        cola_model,  # model being run
        (
            input_sample["input_ids"],
            input_sample["attention_mask"],
        ),  # model input (or a tuple for multiple inputs)
        f"{root_dir}/models/model.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,
        opset_version=10,
        input_names=["input_ids", "attention_mask"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input_ids": {0: "batch_size"},  # variable length axes
            "attention_mask": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    logger.info(
        f"Model converted successfully. ONNX format model is at: {root_dir}/models/model.onnx"
    )


if __name__ == "__main__":
    convert_model()



================================================
FILE: week_5_docker/data.py
================================================
import torch
import datasets
import pytorch_lightning as pl

from datasets import load_dataset
from transformers import AutoTokenizer


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name="google/bert_uncased_L-2_H-128_A-2",
        batch_size=64,
        max_length=128,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self):
        cola_dataset = load_dataset("glue", "cola")
        self.train_data = cola_dataset["train"]
        self.val_data = cola_dataset["validation"]

    def tokenize_data(self, example):
        return self.tokenizer(
            example["sentence"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

    def setup(self, stage=None):
        # we set up only relevant datasets when stage is specified
        if stage == "fit" or stage is None:
            self.train_data = self.train_data.map(self.tokenize_data, batched=True)
            self.train_data.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"]
            )

            self.val_data = self.val_data.map(self.tokenize_data, batched=True)
            self.val_data.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "label"],
                output_all_columns=True,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False
        )


if __name__ == "__main__":
    data_model = DataModule()
    data_model.prepare_data()
    data_model.setup()
    print(next(iter(data_model.train_dataloader()))["input_ids"].shape)



================================================
FILE: week_5_docker/docker-compose.yml
================================================
version: "3"
services:
    prediction_api:
        build: .
        container_name: "inference_container"
        ports:
            - "8000:8000"


================================================
FILE: week_5_docker/Dockerfile
================================================
FROM huggingface/transformers-pytorch-cpu:latest
COPY ./ /app
WORKDIR /app
RUN pip install -r requirements_prod.txt
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]



================================================
FILE: week_5_docker/inference.py
================================================
import torch
from model import ColaModel
from data import DataModule
from utils import timing


class ColaPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = ColaModel.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.freeze()
        self.processor = DataModule()
        self.softmax = torch.nn.Softmax(dim=1)
        self.lables = ["unacceptable", "acceptable"]

    @timing
    def predict(self, text):
        inference_sample = {"sentence": text}
        processed = self.processor.tokenize_data(inference_sample)
        logits = self.model(
            torch.tensor([processed["input_ids"]]),
            torch.tensor([processed["attention_mask"]]),
        )
        scores = self.softmax(logits[0]).tolist()[0]
        predictions = []
        for score, label in zip(scores, self.lables):
            predictions.append({"label": label, "score": score})
        return predictions


if __name__ == "__main__":
    sentence = "The boy is sitting on a bench"
    predictor = ColaPredictor("./models/best-checkpoint.ckpt")
    print(predictor.predict(sentence))
    sentences = ["The boy is sitting on a bench"] * 10
    for sentence in sentences:
        predictor.predict(sentence)



================================================
FILE: week_5_docker/inference_onnx.py
================================================
import numpy as np
import onnxruntime as ort
from scipy.special import softmax

from data import DataModule
from utils import timing


class ColaONNXPredictor:
    def __init__(self, model_path):
        self.ort_session = ort.InferenceSession(model_path)
        self.processor = DataModule()
        self.lables = ["unacceptable", "acceptable"]

    @timing
    def predict(self, text):
        inference_sample = {"sentence": text}
        processed = self.processor.tokenize_data(inference_sample)

        ort_inputs = {
            "input_ids": np.expand_dims(processed["input_ids"], axis=0),
            "attention_mask": np.expand_dims(processed["attention_mask"], axis=0),
        }
        ort_outs = self.ort_session.run(None, ort_inputs)
        scores = softmax(ort_outs[0])[0]
        predictions = []
        for score, label in zip(scores, self.lables):
            predictions.append({"label": label, "score": float(score)})
        print(predictions)
        return predictions


if __name__ == "__main__":
    sentence = "The boy is sitting on a bench"
    predictor = ColaONNXPredictor("./models/model.onnx")
    print(predictor.predict(sentence))
    sentences = ["The boy is sitting on a bench"] * 10
    for sentence in sentences:
        predictor.predict(sentence)



================================================
FILE: week_5_docker/model.py
================================================
import torch
import wandb
import hydra
import numpy as np
import pandas as pd
import torchmetrics
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification
from omegaconf import OmegaConf, DictConfig
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class ColaModel(pl.LightningModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", lr=3e-5):
        super(ColaModel, self).__init__()
        self.save_hyperparameters()

        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        self.num_classes = 2
        self.train_accuracy_metric = torchmetrics.Accuracy()
        self.val_accuracy_metric = torchmetrics.Accuracy()
        self.f1_metric = torchmetrics.F1(num_classes=self.num_classes)
        self.precision_macro_metric = torchmetrics.Precision(
            average="macro", num_classes=self.num_classes
        )
        self.recall_macro_metric = torchmetrics.Recall(
            average="macro", num_classes=self.num_classes
        )
        self.precision_micro_metric = torchmetrics.Precision(average="micro")
        self.recall_micro_metric = torchmetrics.Recall(average="micro")

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        # loss = F.cross_entropy(logits, batch["label"])
        preds = torch.argmax(outputs.logits, 1)
        train_acc = self.train_accuracy_metric(preds, batch["label"])
        self.log("train/loss", outputs.loss, prog_bar=True, on_epoch=True)
        self.log("train/acc", train_acc, prog_bar=True, on_epoch=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        labels = batch["label"]
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        preds = torch.argmax(outputs.logits, 1)

        # Metrics
        valid_acc = self.val_accuracy_metric(preds, labels)
        precision_macro = self.precision_macro_metric(preds, labels)
        recall_macro = self.recall_macro_metric(preds, labels)
        precision_micro = self.precision_micro_metric(preds, labels)
        recall_micro = self.recall_micro_metric(preds, labels)
        f1 = self.f1_metric(preds, labels)

        # Logging metrics
        self.log("valid/loss", outputs.loss, prog_bar=True, on_step=True)
        self.log("valid/acc", valid_acc, prog_bar=True, on_epoch=True)
        self.log("valid/precision_macro", precision_macro, prog_bar=True, on_epoch=True)
        self.log("valid/recall_macro", recall_macro, prog_bar=True, on_epoch=True)
        self.log("valid/precision_micro", precision_micro, prog_bar=True, on_epoch=True)
        self.log("valid/recall_micro", recall_micro, prog_bar=True, on_epoch=True)
        self.log("valid/f1", f1, prog_bar=True, on_epoch=True)
        return {"labels": labels, "logits": outputs.logits}

    def validation_epoch_end(self, outputs):
        labels = torch.cat([x["labels"] for x in outputs])
        logits = torch.cat([x["logits"] for x in outputs])
        preds = torch.argmax(logits, 1)

        ## There are multiple ways to track the metrics
        # 1. Confusion matrix plotting using inbuilt W&B method
        self.logger.experiment.log(
            {
                "conf": wandb.plot.confusion_matrix(
                    probs=logits.numpy(), y_true=labels.numpy()
                )
            }
        )

        # 2. Confusion Matrix plotting using scikit-learn method
        # wandb.log({"cm": wandb.sklearn.plot_confusion_matrix(labels.numpy(), preds)})

        # 3. Confusion Matric plotting using Seaborn
        # data = confusion_matrix(labels.numpy(), preds.numpy())
        # df_cm = pd.DataFrame(data, columns=np.unique(labels), index=np.unique(labels))
        # df_cm.index.name = "Actual"
        # df_cm.columns.name = "Predicted"
        # plt.figure(figsize=(7, 4))
        # plot = sns.heatmap(
        #     df_cm, cmap="Blues", annot=True, annot_kws={"size": 16}
        # )  # font size
        # self.logger.experiment.log({"Confusion Matrix": wandb.Image(plot)})

        # self.logger.experiment.log(
        #     {"roc": wandb.plot.roc_curve(labels.numpy(), logits.numpy())}
        # )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])



================================================
FILE: week_5_docker/requirements.txt
================================================
pytorch-lightning==1.2.10
datasets==1.6.2
transformers==4.5.1
scikit-learn==0.24.2
wandb
torchmetrics
matplotlib
seaborn
hydra-core
omegaconf
hydra_colorlog
fastapi
uvicorn



================================================
FILE: week_5_docker/requirements_inference.txt
================================================
pytorch-lightning==1.2.10
datasets==1.6.2
scikit-learn==0.24.2
hydra-core
omegaconf
hydra_colorlog
onnxruntime
fastapi
uvicorn



================================================
FILE: week_5_docker/train.py
================================================
import torch
import hydra
import wandb
import logging

import pandas as pd
import pytorch_lightning as pl
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from data import DataModule
from model import ColaModel

logger = logging.getLogger(__name__)


class SamplesVisualisationLogger(pl.Callback):
    def __init__(self, datamodule):
        super().__init__()

        self.datamodule = datamodule

    def on_validation_end(self, trainer, pl_module):
        val_batch = next(iter(self.datamodule.val_dataloader()))
        sentences = val_batch["sentence"]

        outputs = pl_module(val_batch["input_ids"], val_batch["attention_mask"])
        preds = torch.argmax(outputs.logits, 1)
        labels = val_batch["label"]

        df = pd.DataFrame(
            {"Sentence": sentences, "Label": labels.numpy(), "Predicted": preds.numpy()}
        )

        wrong_df = df[df["Label"] != df["Predicted"]]
        trainer.logger.experiment.log(
            {
                "examples": wandb.Table(dataframe=wrong_df, allow_mixed_types=True),
                "global_step": trainer.global_step,
            }
        )


@hydra.main(config_path="./configs", config_name="config")
def main(cfg):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    logger.info(f"Using the model: {cfg.model.name}")
    logger.info(f"Using the tokenizer: {cfg.model.tokenizer}")
    cola_data = DataModule(
        cfg.model.tokenizer, cfg.processing.batch_size, cfg.processing.max_length
    )
    cola_model = ColaModel(cfg.model.name)

    root_dir = hydra.utils.get_original_cwd()
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{root_dir}/models",
        filename="best-checkpoint",
        monitor="valid/loss",
        mode="min",
    )

    early_stopping_callback = EarlyStopping(
        monitor="valid/loss", patience=3, verbose=True, mode="min"
    )

    wandb_logger = WandbLogger(project="MLOps Basics", entity="raviraja")
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, SamplesVisualisationLogger(cola_data), early_stopping_callback],
        log_every_n_steps=cfg.training.log_every_n_steps,
        deterministic=cfg.training.deterministic,
        # limit_train_batches=cfg.training.limit_train_batches,
        # limit_val_batches=cfg.training.limit_val_batches,
    )
    trainer.fit(cola_model, cola_data)
    wandb.finish()


if __name__ == "__main__":
    main()



================================================
FILE: week_5_docker/utils.py
================================================
import time
from functools import wraps


def timing(f):
    """Decorator for timing functions
    Usage:
    @timing
    def function(a):
        pass
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print("function:%r took: %2.5f sec" % (f.__name__, end - start))
        return result

    return wrapper



================================================
FILE: week_5_docker/configs/config.yaml
================================================
defaults:
  - model: default
  - processing: default
  - training: default
  - override hydra/job_logging: colorlog
  - override hydra/hydra_logging: colorlog


================================================
FILE: week_5_docker/configs/model/default.yaml
================================================
name: google/bert_uncased_L-2_H-128_A-2             # model used for training the classifier
tokenizer: google/bert_uncased_L-2_H-128_A-2        # tokenizer used for processing the data


================================================
FILE: week_5_docker/configs/processing/default.yaml
================================================
batch_size: 64
max_length: 128


================================================
FILE: week_5_docker/configs/training/default.yaml
================================================
max_epochs: 1
log_every_n_steps: 10
deterministic: true
limit_train_batches: 0.25
limit_val_batches: ${training.limit_train_batches}


================================================
FILE: week_5_docker/dvcfiles/trained_model.dvc
================================================
wdir: ../models
outs:
- md5: c2f5c0a1954209865b9be1945f33ed6e
  size: 17567709
  path: best-checkpoint.ckpt



================================================
FILE: week_5_docker/experimental_notebooks/data_exploration.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
## Imports
"""

import datasets
import pandas as pd

from datasets import load_dataset

"""
## Dataset
"""

cola_dataset = load_dataset('glue', 'cola')
# Output:
#   Reusing dataset glue (/Users/raviraja/.cache/huggingface/datasets/glue/cola/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad)


cola_dataset
# Output:
#   DatasetDict({

#       train: Dataset({

#           features: ['sentence', 'label', 'idx'],

#           num_rows: 8551

#       })

#       validation: Dataset({

#           features: ['sentence', 'label', 'idx'],

#           num_rows: 1043

#       })

#       test: Dataset({

#           features: ['sentence', 'label', 'idx'],

#           num_rows: 1063

#       })

#   })

train_dataset = cola_dataset['train']
val_dataset = cola_dataset['validation']
test_dataset = cola_dataset['test']

len(train_dataset), len(val_dataset), len(test_dataset)
# Output:
#   (8551, 1043, 1063)

train_dataset[0]
# Output:
#   {'idx': 0,

#    'label': 1,

#    'sentence': "Our friends won't buy this analysis, let alone the next one we propose."}

val_dataset[0]
# Output:
#   {'idx': 0,

#    'label': 1,

#    'sentence': 'The sailors rode the breeze clear of the rocks.'}

test_dataset[0]
# Output:
#   {'idx': 0, 'label': -1, 'sentence': 'Bill whistled past the house.'}

train_dataset.features
# Output:
#   {'sentence': Value(dtype='string', id=None),

#    'label': ClassLabel(num_classes=2, names=['unacceptable', 'acceptable'], names_file=None, id=None),

#    'idx': Value(dtype='int32', id=None)}

train_dataset.filter(lambda example: example['label'] == train_dataset.features['label'].str2int('acceptable'))[:5]
# Output:
#   HBox(children=(FloatProgress(value=0.0, max=9.0), HTML(value='')))
#   

#   {'idx': [0, 1, 2, 3, 4],

#    'label': [1, 1, 1, 1, 1],

#    'sentence': ["Our friends won't buy this analysis, let alone the next one we propose.",

#     "One more pseudo generalization and I'm giving up.",

#     "One more pseudo generalization or I'm giving up.",

#     'The more we study verbs, the crazier they get.',

#     'Day by day the facts are getting murkier.']}

train_dataset.filter(lambda example: example['label'] == train_dataset.features['label'].str2int('unacceptable'))[:5]
# Output:
#   HBox(children=(FloatProgress(value=0.0, max=9.0), HTML(value='')))
#   

#   {'idx': [18, 20, 22, 23, 25],

#    'label': [0, 0, 0, 0, 0],

#    'sentence': ['They drank the pub.',

#     'The professor talked us.',

#     'We yelled ourselves.',

#     'We yelled Harry hoarse.',

#     'Harry coughed himself.']}

"""
## Tokenizing
"""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")

train_dataset = cola_dataset['train']
val_dataset = cola_dataset['validation']
test_dataset = cola_dataset['test']

tokenizer
# Output:
#   PreTrainedTokenizerFast(name_or_path='google/bert_uncased_L-2_H-128_A-2', vocab_size=30522, model_max_len=1000000000000000019884624838656, is_fast=True, padding_side='right', special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'})

print(train_dataset[0]['sentence'])
tokenizer(train_dataset[0]['sentence'])
# Output:
#   Our friends won't buy this analysis, let alone the next one we propose.

#   {'input_ids': [101, 2256, 2814, 2180, 1005, 1056, 4965, 2023, 4106, 1010, 2292, 2894, 1996, 2279, 2028, 2057, 16599, 1012, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

tokenizer.decode(tokenizer(train_dataset[0]['sentence'])['input_ids'])
# Output:
#   "[CLS] our friends won't buy this analysis, let alone the next one we propose. [SEP]"

def encode(examples):
    return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

train_dataset = train_dataset.map(encode, batched=True)
# Output:
#   HBox(children=(FloatProgress(value=0.0, max=9.0), HTML(value='')))
#   


"""
## Formatting
"""

import torch

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

"""
## Data Loader
"""

dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

next(iter(dataloader))
# Output:
#   {'attention_mask': tensor([[1, 1, 1,  ..., 0, 0, 0],

#            [1, 1, 1,  ..., 0, 0, 0],

#            [1, 1, 1,  ..., 0, 0, 0],

#            ...,

#            [1, 1, 1,  ..., 0, 0, 0],

#            [1, 1, 1,  ..., 0, 0, 0],

#            [1, 1, 1,  ..., 0, 0, 0]]),

#    'input_ids': tensor([[  101,  2256,  2814,  ...,     0,     0,     0],

#            [  101,  2028,  2062,  ...,     0,     0,     0],

#            [  101,  2028,  2062,  ...,     0,     0,     0],

#            ...,

#            [  101,  5965, 12808,  ...,     0,     0,     0],

#            [  101,  2198, 10948,  ...,     0,     0,     0],

#            [  101,  3021, 24471,  ...,     0,     0,     0]]),

#    'label': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0,

#            1, 0, 0, 1, 1, 1, 1, 1])}

for batch in dataloader:
    print(batch['input_ids'].shape, batch['attention_mask'].shape, batch['label'].shape)
# Output:
#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([7, 512]) torch.Size([7, 512]) torch.Size([7])




================================================
FILE: week_6_github_actions/README.md
================================================

**Note: The purpose of the project to explore the libraries and learn how to use them. Not to build a SOTA model.**

## Requirements:

This project uses Python 3.8

Create a virtual env with the following command:

```
conda create --name project-setup python=3.8
conda activate project-setup
```

Install the requirements:

```
pip install -r requirements.txt
```

## Running

### Training

After installing the requirements, in order to train the model simply run:

```
python train.py
```

### Monitoring

Once the training is completed in the end of the logs you will see something like:

```
wandb: Synced 5 W&B file(s), 4 media file(s), 3 artifact file(s) and 0 other file(s)
wandb:
wandb: Synced proud-mountain-77: https://wandb.ai/raviraja/MLOps%20Basics/runs/3vp1twdc
```

Follow the link to see the wandb dashboard which contains all the plots.

### Versioning data

Refer to the blog: [DVC Configuration](https://www.ravirajag.dev/blog/mlops-dvc)

### Exporting model to ONNX

Once the model is trained, convert the model using the following command:

```
python convert_model_to_onnx.py
```

### Inference

#### Inference using standard pytorch

```
python inference.py
```

#### Inference using ONNX Runtime

```
python inference_onnx.py
```

### Google Service account

Create service account using the steps mentioned here: [Create service account](https://www.ravirajag.dev/blog/mlops-github-actions)

### Configuring dvc

```
dvc init
dvc remote add -d storage gdrive://19JK5AFbqOBlrFVwDHjTrf9uvQFtS0954
dvc remote modify storage gdrive_use_service_account true
dvc remote modify storage gdrive_service_account_json_file_path creds.json
```

`creds.json` is the file created during service account creation


### Docker

Install the docker using the [instructions here](https://docs.docker.com/engine/install/)

Build the image using the command

```shell
docker build -t inference:latest .
```

Then run the container using the command

```shell
docker run -p 8000:8000 --name inference_container inference:latest
```

(or)

Build and run the container using the command

```shell
docker-compose up
```


### Running notebooks

I am using [Jupyter lab](https://jupyter.org/install) to run the notebooks.

Since I am using a virtualenv, when I run the command `jupyter lab` it might or might not use the virtualenv.

To make sure to use the virutalenv, run the following commands before running `jupyter lab`

```
conda install ipykernel
python -m ipykernel install --user --name project-setup
pip install ipywidgets
```


================================================
FILE: week_6_github_actions/app.py
================================================
from fastapi import FastAPI
from inference_onnx import ColaONNXPredictor
app = FastAPI(title="MLOps Basics App")

predictor = ColaONNXPredictor("./models/model.onnx")

@app.get("/")
async def home_page():
    return "<h2>Sample prediction API</h2>"


@app.get("/predict")
async def get_prediction(text: str):
    result =  predictor.predict(text)
    return result


================================================
FILE: week_6_github_actions/convert_model_to_onnx.py
================================================
import torch
import hydra
import logging

from omegaconf.omegaconf import OmegaConf

from model import ColaModel
from data import DataModule

logger = logging.getLogger(__name__)


@hydra.main(config_path="./configs", config_name="config")
def convert_model(cfg):
    root_dir = hydra.utils.get_original_cwd()
    model_path = f"{root_dir}/models/best-checkpoint.ckpt"
    logger.info(f"Loading pre-trained model from: {model_path}")
    cola_model = ColaModel.load_from_checkpoint(model_path)

    data_model = DataModule(
        cfg.model.tokenizer, cfg.processing.batch_size, cfg.processing.max_length
    )
    data_model.prepare_data()
    data_model.setup()
    input_batch = next(iter(data_model.train_dataloader()))
    input_sample = {
        "input_ids": input_batch["input_ids"][0].unsqueeze(0),
        "attention_mask": input_batch["attention_mask"][0].unsqueeze(0),
    }

    # Export the model
    logger.info(f"Converting the model into ONNX format")
    torch.onnx.export(
        cola_model,  # model being run
        (
            input_sample["input_ids"],
            input_sample["attention_mask"],
        ),  # model input (or a tuple for multiple inputs)
        f"{root_dir}/models/model.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,
        opset_version=10,
        input_names=["input_ids", "attention_mask"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input_ids": {0: "batch_size"},  # variable length axes
            "attention_mask": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    logger.info(
        f"Model converted successfully. ONNX format model is at: {root_dir}/models/model.onnx"
    )


if __name__ == "__main__":
    convert_model()



================================================
FILE: week_6_github_actions/data.py
================================================
import torch
import datasets
import pytorch_lightning as pl

from datasets import load_dataset
from transformers import AutoTokenizer


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name="google/bert_uncased_L-2_H-128_A-2",
        batch_size=64,
        max_length=128,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self):
        cola_dataset = load_dataset("glue", "cola")
        self.train_data = cola_dataset["train"]
        self.val_data = cola_dataset["validation"]

    def tokenize_data(self, example):
        return self.tokenizer(
            example["sentence"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

    def setup(self, stage=None):
        # we set up only relevant datasets when stage is specified
        if stage == "fit" or stage is None:
            self.train_data = self.train_data.map(self.tokenize_data, batched=True)
            self.train_data.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"]
            )

            self.val_data = self.val_data.map(self.tokenize_data, batched=True)
            self.val_data.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "label"],
                output_all_columns=True,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False
        )


if __name__ == "__main__":
    data_model = DataModule()
    data_model.prepare_data()
    data_model.setup()
    print(next(iter(data_model.train_dataloader()))["input_ids"].shape)



================================================
FILE: week_6_github_actions/docker-compose.yml
================================================
version: "3"
services:
    prediction_api:
        build: .
        container_name: "inference_container"
        ports:
            - "8000:8000"


================================================
FILE: week_6_github_actions/Dockerfile
================================================
FROM huggingface/transformers-pytorch-cpu:latest

COPY ./ /app
WORKDIR /app

# install requirements
RUN pip install "dvc[gdrive]"
RUN pip install -r requirements_inference.txt

# initialise dvc
RUN dvc init --no-scm
# configuring remote server in dvc
RUN dvc remote add -d storage gdrive://19JK5AFbqOBlrFVwDHjTrf9uvQFtS0954
RUN dvc remote modify storage gdrive_use_service_account true
RUN dvc remote modify storage gdrive_service_account_json_file_path creds.json

RUN cat .dvc/config
# pulling the trained model
RUN dvc pull dvcfiles/trained_model.dvc

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# running the application
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]



================================================
FILE: week_6_github_actions/inference.py
================================================
import torch
from model import ColaModel
from data import DataModule
from utils import timing


class ColaPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = ColaModel.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.freeze()
        self.processor = DataModule()
        self.softmax = torch.nn.Softmax(dim=1)
        self.lables = ["unacceptable", "acceptable"]

    @timing
    def predict(self, text):
        inference_sample = {"sentence": text}
        processed = self.processor.tokenize_data(inference_sample)
        logits = self.model(
            torch.tensor([processed["input_ids"]]),
            torch.tensor([processed["attention_mask"]]),
        )
        scores = self.softmax(logits[0]).tolist()[0]
        predictions = []
        for score, label in zip(scores, self.lables):
            predictions.append({"label": label, "score": score})
        return predictions


if __name__ == "__main__":
    sentence = "The boy is sitting on a bench"
    predictor = ColaPredictor("./models/best-checkpoint.ckpt")
    print(predictor.predict(sentence))
    sentences = ["The boy is sitting on a bench"] * 10
    for sentence in sentences:
        predictor.predict(sentence)



================================================
FILE: week_6_github_actions/inference_onnx.py
================================================
import numpy as np
import onnxruntime as ort
from scipy.special import softmax

from data import DataModule
from utils import timing


class ColaONNXPredictor:
    def __init__(self, model_path):
        self.ort_session = ort.InferenceSession(model_path)
        self.processor = DataModule()
        self.lables = ["unacceptable", "acceptable"]

    @timing
    def predict(self, text):
        inference_sample = {"sentence": text}
        processed = self.processor.tokenize_data(inference_sample)

        ort_inputs = {
            "input_ids": np.expand_dims(processed["input_ids"], axis=0),
            "attention_mask": np.expand_dims(processed["attention_mask"], axis=0),
        }
        ort_outs = self.ort_session.run(None, ort_inputs)
        scores = softmax(ort_outs[0])[0]
        predictions = []
        for score, label in zip(scores, self.lables):
            predictions.append({"label": label, "score": float(score)})
        print(predictions)
        return predictions


if __name__ == "__main__":
    sentence = "The boy is sitting on a bench"
    predictor = ColaONNXPredictor("./models/model.onnx")
    print(predictor.predict(sentence))
    sentences = ["The boy is sitting on a bench"] * 10
    for sentence in sentences:
        predictor.predict(sentence)



================================================
FILE: week_6_github_actions/model.py
================================================
import torch
import wandb
import hydra
import numpy as np
import pandas as pd
import torchmetrics
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification
from omegaconf import OmegaConf, DictConfig
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class ColaModel(pl.LightningModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", lr=3e-5):
        super(ColaModel, self).__init__()
        self.save_hyperparameters()

        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        self.num_classes = 2
        self.train_accuracy_metric = torchmetrics.Accuracy()
        self.val_accuracy_metric = torchmetrics.Accuracy()
        self.f1_metric = torchmetrics.F1(num_classes=self.num_classes)
        self.precision_macro_metric = torchmetrics.Precision(
            average="macro", num_classes=self.num_classes
        )
        self.recall_macro_metric = torchmetrics.Recall(
            average="macro", num_classes=self.num_classes
        )
        self.precision_micro_metric = torchmetrics.Precision(average="micro")
        self.recall_micro_metric = torchmetrics.Recall(average="micro")

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        # loss = F.cross_entropy(logits, batch["label"])
        preds = torch.argmax(outputs.logits, 1)
        train_acc = self.train_accuracy_metric(preds, batch["label"])
        self.log("train/loss", outputs.loss, prog_bar=True, on_epoch=True)
        self.log("train/acc", train_acc, prog_bar=True, on_epoch=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        labels = batch["label"]
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        preds = torch.argmax(outputs.logits, 1)

        # Metrics
        valid_acc = self.val_accuracy_metric(preds, labels)
        precision_macro = self.precision_macro_metric(preds, labels)
        recall_macro = self.recall_macro_metric(preds, labels)
        precision_micro = self.precision_micro_metric(preds, labels)
        recall_micro = self.recall_micro_metric(preds, labels)
        f1 = self.f1_metric(preds, labels)

        # Logging metrics
        self.log("valid/loss", outputs.loss, prog_bar=True, on_step=True)
        self.log("valid/acc", valid_acc, prog_bar=True, on_epoch=True)
        self.log("valid/precision_macro", precision_macro, prog_bar=True, on_epoch=True)
        self.log("valid/recall_macro", recall_macro, prog_bar=True, on_epoch=True)
        self.log("valid/precision_micro", precision_micro, prog_bar=True, on_epoch=True)
        self.log("valid/recall_micro", recall_micro, prog_bar=True, on_epoch=True)
        self.log("valid/f1", f1, prog_bar=True, on_epoch=True)
        return {"labels": labels, "logits": outputs.logits}

    def validation_epoch_end(self, outputs):
        labels = torch.cat([x["labels"] for x in outputs])
        logits = torch.cat([x["logits"] for x in outputs])
        preds = torch.argmax(logits, 1)

        ## There are multiple ways to track the metrics
        # 1. Confusion matrix plotting using inbuilt W&B method
        self.logger.experiment.log(
            {
                "conf": wandb.plot.confusion_matrix(
                    probs=logits.numpy(), y_true=labels.numpy()
                )
            }
        )

        # 2. Confusion Matrix plotting using scikit-learn method
        # wandb.log({"cm": wandb.sklearn.plot_confusion_matrix(labels.numpy(), preds)})

        # 3. Confusion Matric plotting using Seaborn
        # data = confusion_matrix(labels.numpy(), preds.numpy())
        # df_cm = pd.DataFrame(data, columns=np.unique(labels), index=np.unique(labels))
        # df_cm.index.name = "Actual"
        # df_cm.columns.name = "Predicted"
        # plt.figure(figsize=(7, 4))
        # plot = sns.heatmap(
        #     df_cm, cmap="Blues", annot=True, annot_kws={"size": 16}
        # )  # font size
        # self.logger.experiment.log({"Confusion Matrix": wandb.Image(plot)})

        # self.logger.experiment.log(
        #     {"roc": wandb.plot.roc_curve(labels.numpy(), logits.numpy())}
        # )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])



================================================
FILE: week_6_github_actions/parse_json.py
================================================
import json

with open('creds.txt') as f:
	data = f.read()

print(data)
# data = json.loads(data, strict=False)
# print(data)
data = eval(data)
print(data)

with open('test.json', 'w') as f:
	json.dump(data, f)



================================================
FILE: week_6_github_actions/requirements.txt
================================================
pytorch-lightning==1.2.10
datasets==1.6.2
transformers==4.5.1
scikit-learn==0.24.2
wandb
torchmetrics
matplotlib
seaborn
hydra-core
omegaconf
hydra_colorlog
fastapi
uvicorn



================================================
FILE: week_6_github_actions/requirements_inference.txt
================================================
pytorch-lightning==1.2.10
datasets==1.6.2
scikit-learn==0.24.2
hydra-core
omegaconf
hydra_colorlog
onnxruntime
fastapi
uvicorn
dvc


================================================
FILE: week_6_github_actions/train.py
================================================
import torch
import hydra
import wandb
import logging

import pandas as pd
import pytorch_lightning as pl
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from data import DataModule
from model import ColaModel

logger = logging.getLogger(__name__)


class SamplesVisualisationLogger(pl.Callback):
    def __init__(self, datamodule):
        super().__init__()

        self.datamodule = datamodule

    def on_validation_end(self, trainer, pl_module):
        val_batch = next(iter(self.datamodule.val_dataloader()))
        sentences = val_batch["sentence"]

        outputs = pl_module(val_batch["input_ids"], val_batch["attention_mask"])
        preds = torch.argmax(outputs.logits, 1)
        labels = val_batch["label"]

        df = pd.DataFrame(
            {"Sentence": sentences, "Label": labels.numpy(), "Predicted": preds.numpy()}
        )

        wrong_df = df[df["Label"] != df["Predicted"]]
        trainer.logger.experiment.log(
            {
                "examples": wandb.Table(dataframe=wrong_df, allow_mixed_types=True),
                "global_step": trainer.global_step,
            }
        )


@hydra.main(config_path="./configs", config_name="config")
def main(cfg):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    logger.info(f"Using the model: {cfg.model.name}")
    logger.info(f"Using the tokenizer: {cfg.model.tokenizer}")
    cola_data = DataModule(
        cfg.model.tokenizer, cfg.processing.batch_size, cfg.processing.max_length
    )
    cola_model = ColaModel(cfg.model.name)

    root_dir = hydra.utils.get_original_cwd()
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{root_dir}/models",
        filename="best-checkpoint",
        monitor="valid/loss",
        mode="min",
    )

    early_stopping_callback = EarlyStopping(
        monitor="valid/loss", patience=3, verbose=True, mode="min"
    )

    wandb_logger = WandbLogger(project="MLOps Basics", entity="raviraja")
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, SamplesVisualisationLogger(cola_data), early_stopping_callback],
        log_every_n_steps=cfg.training.log_every_n_steps,
        deterministic=cfg.training.deterministic,
        # limit_train_batches=cfg.training.limit_train_batches,
        # limit_val_batches=cfg.training.limit_val_batches,
    )
    trainer.fit(cola_model, cola_data)
    wandb.finish()


if __name__ == "__main__":
    main()



================================================
FILE: week_6_github_actions/utils.py
================================================
import time
from functools import wraps


def timing(f):
    """Decorator for timing functions
    Usage:
    @timing
    def function(a):
        pass
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print("function:%r took: %2.5f sec" % (f.__name__, end - start))
        return result

    return wrapper



================================================
FILE: week_6_github_actions/configs/config.yaml
================================================
defaults:
  - model: default
  - processing: default
  - training: default
  - override hydra/job_logging: colorlog
  - override hydra/hydra_logging: colorlog


================================================
FILE: week_6_github_actions/configs/model/default.yaml
================================================
name: google/bert_uncased_L-2_H-128_A-2             # model used for training the classifier
tokenizer: google/bert_uncased_L-2_H-128_A-2        # tokenizer used for processing the data


================================================
FILE: week_6_github_actions/configs/processing/default.yaml
================================================
batch_size: 64
max_length: 128


================================================
FILE: week_6_github_actions/configs/training/default.yaml
================================================
max_epochs: 1
log_every_n_steps: 10
deterministic: true
limit_train_batches: 0.25
limit_val_batches: ${training.limit_train_batches}


================================================
FILE: week_6_github_actions/dvcfiles/trained_model.dvc
================================================
wdir: ../models
outs:
- md5: d82b8390fa2f09b121de4abfa094a7a9
  size: 17562590
  path: model.onnx



================================================
FILE: week_6_github_actions/experimental_notebooks/data_exploration.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
## Imports
"""

import datasets
import pandas as pd

from datasets import load_dataset

"""
## Dataset
"""

cola_dataset = load_dataset('glue', 'cola')
# Output:
#   Reusing dataset glue (/Users/raviraja/.cache/huggingface/datasets/glue/cola/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad)


cola_dataset
# Output:
#   DatasetDict({

#       train: Dataset({

#           features: ['sentence', 'label', 'idx'],

#           num_rows: 8551

#       })

#       validation: Dataset({

#           features: ['sentence', 'label', 'idx'],

#           num_rows: 1043

#       })

#       test: Dataset({

#           features: ['sentence', 'label', 'idx'],

#           num_rows: 1063

#       })

#   })

train_dataset = cola_dataset['train']
val_dataset = cola_dataset['validation']
test_dataset = cola_dataset['test']

len(train_dataset), len(val_dataset), len(test_dataset)
# Output:
#   (8551, 1043, 1063)

train_dataset[0]
# Output:
#   {'idx': 0,

#    'label': 1,

#    'sentence': "Our friends won't buy this analysis, let alone the next one we propose."}

val_dataset[0]
# Output:
#   {'idx': 0,

#    'label': 1,

#    'sentence': 'The sailors rode the breeze clear of the rocks.'}

test_dataset[0]
# Output:
#   {'idx': 0, 'label': -1, 'sentence': 'Bill whistled past the house.'}

train_dataset.features
# Output:
#   {'sentence': Value(dtype='string', id=None),

#    'label': ClassLabel(num_classes=2, names=['unacceptable', 'acceptable'], names_file=None, id=None),

#    'idx': Value(dtype='int32', id=None)}

train_dataset.filter(lambda example: example['label'] == train_dataset.features['label'].str2int('acceptable'))[:5]
# Output:
#   HBox(children=(FloatProgress(value=0.0, max=9.0), HTML(value='')))
#   

#   {'idx': [0, 1, 2, 3, 4],

#    'label': [1, 1, 1, 1, 1],

#    'sentence': ["Our friends won't buy this analysis, let alone the next one we propose.",

#     "One more pseudo generalization and I'm giving up.",

#     "One more pseudo generalization or I'm giving up.",

#     'The more we study verbs, the crazier they get.',

#     'Day by day the facts are getting murkier.']}

train_dataset.filter(lambda example: example['label'] == train_dataset.features['label'].str2int('unacceptable'))[:5]
# Output:
#   HBox(children=(FloatProgress(value=0.0, max=9.0), HTML(value='')))
#   

#   {'idx': [18, 20, 22, 23, 25],

#    'label': [0, 0, 0, 0, 0],

#    'sentence': ['They drank the pub.',

#     'The professor talked us.',

#     'We yelled ourselves.',

#     'We yelled Harry hoarse.',

#     'Harry coughed himself.']}

"""
## Tokenizing
"""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")

train_dataset = cola_dataset['train']
val_dataset = cola_dataset['validation']
test_dataset = cola_dataset['test']

tokenizer
# Output:
#   PreTrainedTokenizerFast(name_or_path='google/bert_uncased_L-2_H-128_A-2', vocab_size=30522, model_max_len=1000000000000000019884624838656, is_fast=True, padding_side='right', special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'})

print(train_dataset[0]['sentence'])
tokenizer(train_dataset[0]['sentence'])
# Output:
#   Our friends won't buy this analysis, let alone the next one we propose.

#   {'input_ids': [101, 2256, 2814, 2180, 1005, 1056, 4965, 2023, 4106, 1010, 2292, 2894, 1996, 2279, 2028, 2057, 16599, 1012, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

tokenizer.decode(tokenizer(train_dataset[0]['sentence'])['input_ids'])
# Output:
#   "[CLS] our friends won't buy this analysis, let alone the next one we propose. [SEP]"

def encode(examples):
    return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

train_dataset = train_dataset.map(encode, batched=True)
# Output:
#   HBox(children=(FloatProgress(value=0.0, max=9.0), HTML(value='')))
#   


"""
## Formatting
"""

import torch

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

"""
## Data Loader
"""

dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

next(iter(dataloader))
# Output:
#   {'attention_mask': tensor([[1, 1, 1,  ..., 0, 0, 0],

#            [1, 1, 1,  ..., 0, 0, 0],

#            [1, 1, 1,  ..., 0, 0, 0],

#            ...,

#            [1, 1, 1,  ..., 0, 0, 0],

#            [1, 1, 1,  ..., 0, 0, 0],

#            [1, 1, 1,  ..., 0, 0, 0]]),

#    'input_ids': tensor([[  101,  2256,  2814,  ...,     0,     0,     0],

#            [  101,  2028,  2062,  ...,     0,     0,     0],

#            [  101,  2028,  2062,  ...,     0,     0,     0],

#            ...,

#            [  101,  5965, 12808,  ...,     0,     0,     0],

#            [  101,  2198, 10948,  ...,     0,     0,     0],

#            [  101,  3021, 24471,  ...,     0,     0,     0]]),

#    'label': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0,

#            1, 0, 0, 1, 1, 1, 1, 1])}

for batch in dataloader:
    print(batch['input_ids'].shape, batch['attention_mask'].shape, batch['label'].shape)
# Output:
#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([32, 512]) torch.Size([32, 512]) torch.Size([32])

#   torch.Size([7, 512]) torch.Size([7, 512]) torch.Size([7])




================================================
FILE: week_7_ecr/README.md
================================================

**Note: The purpose of the project to explore the libraries and learn how to use them. Not to build a SOTA model.**

## Requirements:

This project uses Python 3.8

Create a virtual env with the following command:

```
conda create --name project-setup python=3.8
conda activate project-setup
```

Install the requirements:

```
pip install -r requirements.txt
```

## Running

### Training

After installing the requirements, in order to train the model simply run:

```
python train.py
```

### Monitoring

Once the training is completed in the end of the logs you will see something like:

```
wandb: Synced 5 W&B file(s), 4 media file(s), 3 artifact file(s) and 0 other file(s)
wandb:
wandb: Synced proud-mountain-77: https://wandb.ai/raviraja/MLOps%20Basics/runs/3vp1twdc
```

Follow the link to see the wandb dashboard which contains all the plots.

### Versioning data

Refer to the blog: [DVC Configuration](https://www.ravirajag.dev/blog/mlops-dvc)

### Exporting model to ONNX

Once the model is trained, convert the model using the following command:

```
python convert_model_to_onnx.py
```

### Inference

#### Inference using standard pytorch

```
python inference.py
```

#### Inference using ONNX Runtime

```
python inference_onnx.py
```

## S3 & ECR

Follow the instructions mentioned in the [blog post](https://www.ravirajag.dev/blog/mlops-container-registry) for creating S3 bucket and ECR repository. 

### Configuring dvc

```
dvc init (this has to be done at root folder)
dvc remote add -d model-store s3://models-dvc/trained_models/
```

### AWS credentials

Create the credentials as mentioned in the [blog post](https://www.ravirajag.dev/blog/mlops-container-registry)

**Do not share the secrets with others**

Set the ACCESS key and id values in environment variables.

```
export AWS_ACCESS_KEY_ID=<ACCESS KEY ID>
export AWS_SECRET_ACCESS_KEY=<ACCESS SECRET>
```

### Trained model in DVC

Sdd the trained model(onnx) to dvc using the following command:

```shell
cd dvcfiles
dvc add ../models/model.onnx --file trained_model.dvc
```

Push the model to remote storage

```shell
dvc push trained_model.dvc
```

### Docker

Install the docker using the [instructions here](https://docs.docker.com/engine/install/)

Build the image using the command

```shell
docker build -t mlops-basics:latest .
```

Then run the container using the command

```shell
docker run -p 8000:8000 --name inference_container mlops-basics:latest
```

(or)

Build and run the container using the command

```shell
docker-compose up
```

### Pushing the image to ECR

Follow the instructions mentioned in [blog post](https://www.ravirajag.dev/blog/mlops-container-registry) for creating ECR repository.

- Authenticating docker client to ECR

```
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 246113150184.dkr.ecr.us-west-2.amazonaws.com
```

- Tagging the image

```
docker tag mlops-basics:latest 246113150184.dkr.ecr.us-west-2.amazonaws.com/mlops-basics:latest
```

- Pushing the image

```
docker push 246113150184.dkr.ecr.us-west-2.amazonaws.com/mlops-basics:latest
```

Refer to `.github/workflows/build_docker_image.yaml` file for automatically creating the docker image with trained model and pushing it to ECR.


### Running notebooks

I am using [Jupyter lab](https://jupyter.org/install) to run the notebooks.

Since I am using a virtualenv, when I run the command `jupyter lab` it might or might not use the virtualenv.

To make sure to use the virutalenv, run the following commands before running `jupyter lab`

```
conda install ipykernel
python -m ipykernel install --user --name project-setup
pip install ipywidgets
```


================================================
FILE: week_7_ecr/app.py
================================================
from fastapi import FastAPI
from inference_onnx import ColaONNXPredictor
app = FastAPI(title="MLOps Basics App")

predictor = ColaONNXPredictor("./models/model.onnx")

@app.get("/")
async def home_page():
    return "<h2>Sample prediction API</h2>"


@app.get("/predict")
async def get_prediction(text: str):
    result =  predictor.predict(text)
    return result


================================================
FILE: week_7_ecr/convert_model_to_onnx.py
================================================
import torch
import hydra
import logging

from omegaconf.omegaconf import OmegaConf

from model import ColaModel
from data import DataModule

logger = logging.getLogger(__name__)


@hydra.main(config_path="./configs", config_name="config")
def convert_model(cfg):
    root_dir = hydra.utils.get_original_cwd()
    model_path = f"{root_dir}/models/best-checkpoint.ckpt"
    logger.info(f"Loading pre-trained model from: {model_path}")
    cola_model = ColaModel.load_from_checkpoint(model_path)

    data_model = DataModule(
        cfg.model.tokenizer, cfg.processing.batch_size, cfg.processing.max_length
    )
    data_model.prepare_data()
    data_model.setup()
    input_batch = next(iter(data_model.train_dataloader()))
    input_sample = {
        "input_ids": input_batch["input_ids"][0].unsqueeze(0),
        "attention_mask": input_batch["attention_mask"][0].unsqueeze(0),
    }

    # Export the model
    logger.info(f"Converting the model into ONNX format")
    torch.onnx.export(
        cola_model,  # model being run
        (
            input_sample["input_ids"],
            input_sample["attention_mask"],
        ),  # model input (or a tuple for multiple inputs)
        f"{root_dir}/models/model.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,
        opset_version=10,
        input_names=["input_ids", "attention_mask"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input_ids": {0: "batch_size"},  # variable length axes
            "attention_mask": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    logger.info(
        f"Model converted successfully. ONNX format model is at: {root_dir}/models/model.onnx"
    )


if __name__ == "__main__":
    convert_model()



================================================
FILE: week_7_ecr/data.py
================================================
import torch
import datasets
import pytorch_lightning as pl

from datasets import load_dataset
from transformers import AutoTokenizer


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name="google/bert_uncased_L-2_H-128_A-2",
        batch_size=64,
        max_length=128,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self):
        cola_dataset = load_dataset("glue", "cola")
        self.train_data = cola_dataset["train"]
        self.val_data = cola_dataset["validation"]

    def tokenize_data(self, example):
        return self.tokenizer(
            example["sentence"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

    def setup(self, stage=None):
        # we set up only relevant datasets when stage is specified
        if stage == "fit" or stage is None:
            self.train_data = self.train_data.map(self.tokenize_data, batched=True)
            self.train_data.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"]
            )

            self.val_data = self.val_data.map(self.tokenize_data, batched=True)
            self.val_data.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "label"],
                output_all_columns=True,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False
        )


if __name__ == "__main__":
    data_model = DataModule()
    data_model.prepare_data()
    data_model.setup()
    print(next(iter(data_model.train_dataloader()))["input_ids"].shape)



================================================
FILE: week_7_ecr/docker-compose.yml
================================================
version: "3"
services:
    prediction_api:
        build: .
        container_name: "inference_container"
        ports:
            - "8000:8000"


================================================
FILE: week_7_ecr/Dockerfile
================================================
FROM huggingface/transformers-pytorch-cpu:latest

COPY ./ /app
WORKDIR /app

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY


#this envs are experimental
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY


# install requirements
RUN pip install "dvc[s3]"
RUN pip install -r requirements_inference.txt

# initialise dvc
RUN dvc init --no-scm
# configuring remote server in dvc
RUN dvc remote add -d model-store s3://models-dvc/trained_models/

RUN cat .dvc/config
# pulling the trained model
RUN dvc pull dvcfiles/trained_model.dvc

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# running the application
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]



================================================
FILE: week_7_ecr/inference.py
================================================
import torch
from model import ColaModel
from data import DataModule
from utils import timing


class ColaPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = ColaModel.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.freeze()
        self.processor = DataModule()
        self.softmax = torch.nn.Softmax(dim=1)
        self.lables = ["unacceptable", "acceptable"]

    @timing
    def predict(self, text):
        inference_sample = {"sentence": text}
        processed = self.processor.tokenize_data(inference_sample)
        logits = self.model(
            torch.tensor([processed["input_ids"]]),
            torch.tensor([processed["attention_mask"]]),
        )
        scores = self.softmax(logits[0]).tolist()[0]
        predictions = []
        for score, label in zip(scores, self.lables):
            predictions.append({"label": label, "score": score})
        return predictions


if __name__ == "__main__":
    sentence = "The boy is sitting on a bench"
    predictor = ColaPredictor("./models/best-checkpoint.ckpt")
    print(predictor.predict(sentence))
    sentences = ["The boy is sitting on a bench"] * 10
    for sentence in sentences:
        predictor.predict(sentence)



================================================
FILE: week_7_ecr/inference_onnx.py
================================================
import numpy as np
import onnxruntime as ort
from scipy.special import softmax

from data import DataModule
from utils import timing


class ColaONNXPredictor:
    def __init__(self, model_path):
        self.ort_session = ort.InferenceSession(model_path)
        self.processor = DataModule()
        self.lables = ["unacceptable", "acceptable"]

    @timing
    def predict(self, text):
        inference_sample = {"sentence": text}
        processed = self.processor.tokenize_data(inference_sample)

        ort_inputs = {
            "input_ids": np.expand_dims(processed["input_ids"], axis=0),
            "attention_mask": np.expand_dims(processed["attention_mask"], axis=0),
        }
        ort_outs = self.ort_session.run(None, ort_inputs)
        scores = softmax(ort_outs[0])[0]
        predictions = []
        for score, label in zip(scores, self.lables):
            predictions.append({"label": label, "score": float(score)})
        print(predictions)
        return predictions


if __name__ == "__main__":
    sentence = "The boy is sitting on a bench"
    predictor = ColaONNXPredictor("./models/model.onnx")
    print(predictor.predict(sentence))
    sentences = ["The boy is sitting on a bench"] * 10
    for sentence in sentences:
        predictor.predict(sentence)



================================================
FILE: week_7_ecr/model.py
================================================
import torch
import wandb
import hydra
import numpy as np
import pandas as pd
import torchmetrics
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification
from omegaconf import OmegaConf, DictConfig
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class ColaModel(pl.LightningModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", lr=3e-5):
        super(ColaModel, self).__init__()
        self.save_hyperparameters()

        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        self.num_classes = 2
        self.train_accuracy_metric = torchmetrics.Accuracy()
        self.val_accuracy_metric = torchmetrics.Accuracy()
        self.f1_metric = torchmetrics.F1(num_classes=self.num_classes)
        self.precision_macro_metric = torchmetrics.Precision(
            average="macro", num_classes=self.num_classes
        )
        self.recall_macro_metric = torchmetrics.Recall(
            average="macro", num_classes=self.num_classes
        )
        self.precision_micro_metric = torchmetrics.Precision(average="micro")
        self.recall_micro_metric = torchmetrics.Recall(average="micro")

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        # loss = F.cross_entropy(logits, batch["label"])
        preds = torch.argmax(outputs.logits, 1)
        train_acc = self.train_accuracy_metric(preds, batch["label"])
        self.log("train/loss", outputs.loss, prog_bar=True, on_epoch=True)
        self.log("train/acc", train_acc, prog_bar=True, on_epoch=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        labels = batch["label"]
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        preds = torch.argmax(outputs.logits, 1)

        # Metrics
        valid_acc = self.val_accuracy_metric(preds, labels)
        precision_macro = self.precision_macro_metric(preds, labels)
       