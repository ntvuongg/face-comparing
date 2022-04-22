# Face Comparing Project 😎
***
✍ An simple face comparing app using [**VGG-16**](https://keras.io/api/applications/vgg/#vgg16-function) to extract features, compare two photos and see if they are the same person.

📌 **Project pipeline:**

> 1. Get input (images) from user.

> 2. Using **MTCNN** to detect face.

> 3. Processing face detected in step 2.

> 4. Using **VGG-16** to extract features as vectors.

> 5. Evaluating 2 vectors using **Cosine Similarity** or **Euclidean Distance (L2 Norm)**.

> 6. Building User Interface (UI) by using [**Streamlit**](https://streamlit.io/).

❗ **Note:** Uploaded images should follow these bellow:

> ☞  Image contains only **01 face**.

> ☞  Image of face should be **frontal** and **center aligned**.

> ☞  Image maximum size supported: **200MB**

⌛ This is not final version! Source code will be updated in the future.

👋 Have fun!