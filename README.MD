# Amazon SageMaker Solution for Privacy in Natural Language Processing

More and more text data are becoming available these days to train Natural Language Processing models such as
sentiment analysis, predictive keyboards and question-answering chatbots. If companies that deploy such models use data provided by users, they have a responsibility to take steps to ensure their users' privacy.

In this solution we demonstrate how one can use Differential Privacy to build accurate
natural language processing models to meet business goals, while helping protect the privacy of the individuals in
the data.

We will be using an algorithm,
created by Amazon scientists, [Feyisetan et al.]((https://www.amazon.science/blog/preserving-privacy-in-analyses-of-textual-data)), that customers can use to better preserve the privact of their users when analyzing
large-scale text data.


## Getting Started with a SageMaker Notebook Instance

If launching the solution using the included CloudFormation template, use the following quick-launch link to create a CloudFormation stack and
deploy the resources in this project.

<table>
  <tr>
    <th colspan="3">AWS Region</td>
    <th>AWS CloudFormation</td>
  </tr>
  <tr>
    <td>US West</td>
    <td>Oregon</td>
    <td>us-west-2</td>
    <td align="center">
      <a href="https://us-west-2.console.aws.amazon.com/cloudformation/home?region=us-west-2#/stacks/create/review?templateURL=https://sagemaker-solutions-prod-us-west-2.s3-us-west-2.amazonaws.com/sagemaker-privacy-for-nlp/deployment/sagemaker-privacy-for-nlp.yaml&stackName=sagemaker-privacy-for-nlp">
        <img src="docs/launch_button.svg" height="30">
      </a>
    </td>
  </tr>
</table>

On the stack creation page, check the box to acknowledge creation of IAM resources,
and click Create Stack.


Once stack creation has completed successfully,
click the 'NotebookSourceCode' link on the 'Outputs' tab. Click '1. Data Privatization.ipynb' and follow the instruction inside the notebook.

**Caution**: Cloning this GitHub repository and running the code manually could lead to unexpected issues! We recommend you use the AWS
CloudFormation template. You'll get an Amazon SageMaker Notebook instance that's been correctly setup and configured to access the other resources in the solution.


## Architecture

<img src="docs/architecture.png" alt="Architecture Diagram" width="600"/>

## Contents

* `deployment/`
    * `sagemaker-privacy-for-nlp.yaml`: The CloudFormation template that is used to launch the solution.
* `docs/`: Contains the images and figures used in the README and notebooks.
* `source/sagemaker/`:
    * `package/`
        * `config.py`: Stores and retrieves project configuration.
        * `utils.py`: Utility functions for scripts and/or notebooks.
        * `data_privatization/`
            * `container/`: Contains the Docker container definition for running the privatization on one worker.
            * `distributed-container/`: Contains the Docker container definition for running the privatization on a cluster with multiple workers.
            * `data_privatization.py`: The code used to apply the privatization mechanism.
        * `model/`
            * `inference.py`: Contains the inference code used to evaluate the trained models.
            * `requirements.txt`: A file that defines the libraries needed to execute the privatization and inference using Amazon SageMaker Processing.
            * `train.py`: Contains the code used to train a model using Torchtext in SageMaker Training.
    * `1. Data Privatization.ipynb`: The first notebook, used to demonstrate and run the privatization mechanism.
    * `2. Model Training.ipynb`: The second notebook, used to train and compare two models trained on sensitive and privatized data, that were generated using the first notebook.
    * `requirements.txt`: A file that defines the libraries needed to execute the notebooks.
    * `setup.py`: Definition file of the package used in the solution.


## FAQ

### What is differential privacy?

Differential Privacy is a formalized notion of privacy that helps protect the
privacy of individuals in a database against attackers who try to re-identify them.
It was originally proposed by [Dwork at al.](https://www.semanticscholar.org/paper/Calibrating-Noise-to-Sensitivity-in-Private-Data-Dwork-McSherry/e4ce10063cd25447dcde75c2d9ce327446ced952)
and has since become very widely across both industry and academia to help preserve the
privacy of individuals in databases. Amongst many other deployments, the [2020 US census](https://www.ncsl.org/research/redistricting/differential-privacy-for-census-data-explained.aspx)
is using differential privacy so that the data "from individuals and individual
households remain confidential".

### How does differential privacy work?

Differential Privacy (DP) works by introducing a carefully designed amount of noise
into the results of queries to a database. For example, a query to a database
could be "what is the number of individuals with an income above $45,000 who reside
in Seattle, WA".

Every time such a query is executed, DP will inject some noise into the result
and return the noisy result.
Specifically, DP  provides a probabilistic guarantee that given two datasets that only differ in the presence
of a single individual in the datasets, the probability that the outcome of a query was produced from either dataset is
roughly equal.

In other words, regardless of the query, an attacker cannot be certain that
any single individual's data were used in the calculation of the result.

### Why do we need differential privacy for Natural Language Processing?

Machine Learning models work by learning to associate inputs to outputs from large data
sources. If the data sources contain sensitive data, it may be possible to use the trained
model to extract information about the individuals whose data were used to train
the machine learning model.

Researchers have been able to perform _membership inference attacks_ on black-box models,
without having access to the data or model internals, by querying the public-facing
API of a model.
The researchers were able to reveal information about individuals' presence in sensitive datasets, such as a [hospital
discharge dataset](https://arxiv.org/abs/1610.05820), or infer whether a [particular
users' text data](https://arxiv.org/abs/1811.00513) were used to train an NLP model.

This illustrates the need to implement privacy measures when one makes a model
trained on sensitive user data available to a wider audience.


### What is the algorithm we will use for privacy for Natural Language Processing?

The privatization algorithm we will use, was [designed by Amazon's scientists](https://www.amazon.science/blog/preserving-privacy-in-analyses-of-textual-data)
as "a way to protect privacy during large-scale analyses of textual data supplied by customers".

It uses a variant of differential privacy known as "metric" differential privacy that works
in metric spaces, such as the space of word embeddings. The idea of metric DP originated
in privacy research for location services. Given a query such as, "show me restaurants
around my location" the purpose is to give useful results to the user, without revealing
their exact location to the service. The idea is again to add carefully crafted noise to
the exact location of the user, and return results for the new noisy location.

This algorithm takes this idea and adapts it to the space of word embeddings. Given an input word,
we want to _perturb_ the input in a way that preserves the meaning or _semantics_
of the word, while providing privacy for the user. It will inject noise to the embedding vector
of the original word, then pick another word close to the resulting noisy vector to replace the
original.

<img src="source/sagemaker/images/privatization-example.png" alt="Example of noise injection" width="600"/>

In the example above, the input word is "Phone". We inject noise to move away from the original
word in the embedding space, and choose a new word whose position is closest to the new noisy vector, in this case
"Mobile". This way we have preserved the meaning of the original word, while at the same time
provided better preserved the privacy of the individuals in the model: if an attacker tries
to reverse engineer a downstream model, they cannot be certain whether a specific word or sequence originated
from an individual, or if it is the result of a perturbation.


### Resources

#### Text-privact algorithm explanation and paper
* [Blog post](https://www.amazon.science/blog/preserving-privacy-in-analyses-of-textual-data)
* [Paper](https://arxiv.org/abs/1910.08902)

#### Differential Privacy Intuition
* [Differential Privacy: A Primer for a Non-Technical Audience](https://dash.harvard.edu/handle/1/38323292)
* [Differential privacy, an easy case](https://accuracyandprivacy.substack.com/)

## Aknowledgements

This solution uses a subset of the _Amazon Reviews - Polarity_ dataset, created by Xiang et al. [1] which itself is
based on the Amazon Reviews Dataset by McAuley et al. [2]. The data are publically available and hosted
on the AWS Data Registry [3], on the behalf of fast.ai [4].

1. Xiang Zhang, Junbo Zhao, and Yann LeCun. 2015. Character-level convolutional networks for text classification. In Proceedings of the 28th International Conference on Neural Information Processing Systems - Volume 1 (NIPS’15). MIT Press, Cambridge, MA, USA, 649–657.
2. J. McAuley and J. Leskovec.  Hidden factors and hidden topics: Understanding rating dimensions with review text. In Proceedings of the 7th ACM Conference on Recommender Systems, RecSys ’13, pages 165–172, New York, NY, USA, 2013. ACM.
3. https://registry.opendata.aws/fast-ai-nlp/
4. https://course.fast.ai/datasets#nlp

The solution also uses the pre-trained GloVe.6B vectors, created by [Stanford NLP](https://nlp.stanford.edu/projects/glove/) and released under the [Open Data Commons Public Domain Dedication and License](https://opendatacommons.org/licenses/pddl/). The original publication is cited below:

Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation. [pdf](https://nlp.stanford.edu/pubs/glove.pdf)

## License

This project is licensed under the Amazon Software License.