
# Bonus Assignment - CS5720 Neural Networks and Deep Learning

##  Student Info
- **Name:** Shreya Surakanti
- **Course:** CS5720 - Neural Networks and Deep Learning

---

##  Setup Instructions

###  Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
````

###  Install Required Libraries

For both parts of the assignment:

```bash
pip install torch torchvision transformers
```

---

##  How to Run the Code

###  Question 1: Question Answering with Transformers

Run basic pipeline (default model):



1. **Basic Pipeline (Default Model)**

```bash
python basic_pipeline.py
```

2. **Using Custom Pretrained Model (deepset/roberta-base-squad2)**

```bash
python custom_model.py
```

3. **Test on Custom Example (Your Own Context & Questions)**

```bash
python my_examples.py
```

---

###  Question 2: Conditional GAN for MNIST Digit Generation

Run the following script to train the Conditional GAN and generate images:


```bash
python cgan_mnist.py
```

* Output images will be saved after each epoch in the `generated_images/` folder.
* Each image will contain digits from `0` to `9`, one per column.

---

##  Explanation of Work

### Question 1: Transformers-Based Question Answering

* Implemented a basic QA system using Hugging Face’s `transformers` pipeline.
* First used the default model to extract answers from a given context.
* Then replaced it with a custom pretrained model (`deepset/roberta-base-squad2`) to improve accuracy.
* Finally, tested with self-written context and two different questions to verify relevance and performance.

### Question 2: Conditional GAN for MNIST

* Built a Conditional GAN that generates images of digits 0–9 based on input labels.
* Modified both Generator and Discriminator to accept digit labels along with input.
* Generated a row of 10 digit images, each corresponding to labels from 0 to 9.
* Output shows improvement in digit clarity as training progresses.

---

##  Short Answer Section

**1. How does a Conditional GAN differ from a vanilla GAN?**
Conditional GANs are guided by labels (like digits or categories) to control output, while vanilla GANs generate random outputs without context.
 *Example:* Generating images of a particular digit, class, or facial expression.

**2. What does the discriminator learn in an image-to-image GAN? Why is pairing important?**
The discriminator learns the mapping relationship between input and output images (not just image realism).
Pairing is important to help the model evaluate whether the generated image correctly corresponds to the input.
