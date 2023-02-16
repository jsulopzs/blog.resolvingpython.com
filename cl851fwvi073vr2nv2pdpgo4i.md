# The Resolving Python Method

We all feel lost and frustrated when we don't know the code to solve a problem in Python.

❌ We lack a **framework** to solve problems

✅ Introducing the *Resolving Python Method*, a framework you can master to reach your solutions faster without even looking at Google.

Stick to the following statement every time you need code to solve a problem:

Programming is nothing more than applying functions to objects to transform them into other objects.

```
Function (Object A) -> Object B
```

For example, to develop a Machine Learning model, you pass the DataFrame (`Object A`) to fit (`Function`) the Mathematical Equation (`Object B`).

Even though functions are inside libraries, they don't perform the algorithms (libraries just store functions). Therefore, your first approach should be: 

Which `Function()` do we need to transform `Object A` into `Object B`?

In Python, we should know we have three ways to access functions:

1. `object.function()`
2. `library.function()`
3. `built_in_function()`

## Function within the Object

**Most of the functions we use in a Python script come from the Object.**

If we press the `[tab]` key after the dot `.`, Python suggests a list of the functions we can use from the Object.

```
object. + [tab] 
```

%[https://youtu.be/EA0y6D5zFgk]

## Function within the Library

If the function we are looking for doesn't appear in the list, we think about the library in which the function might be.

```
import library

library. + [tab]
```

Each library contains functions of a specific topic.

- NumPy: Mathematical Operations with Numbers
- Scikit-Learn: Machine Learning algorithms
- Matplotlib: Mathematical Plots
- Seaborn: Plots
- Pandas: Data Analysis

%[https://youtu.be/yP7tRUIs3jw]

