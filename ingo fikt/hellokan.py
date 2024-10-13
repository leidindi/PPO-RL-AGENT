#!/usr/bin/env python
# coding: utf-8

# # Hello, KAN!

# ### Kolmogorov-Arnold representation theorem

# Kolmogorov-Arnold representation theorem states that if $f$ is a multivariate continuous function
# on a bounded domain, then it can be written as a finite composition of continuous functions of a
# single variable and the binary operation of addition. More specifically, for a smooth $f : [0,1]^n \to \mathbb{R}$,
# 
# 
# $$f(x) = f(x_1,...,x_n)=\sum_{q=1}^{2n+1}\Phi_q(\sum_{p=1}^n \phi_{q,p}(x_p))$$
# 
# where $\phi_{q,p}:[0,1]\to\mathbb{R}$ and $\Phi_q:\mathbb{R}\to\mathbb{R}$. In a sense, they showed that the only true multivariate function is addition, since every other function can be written using univariate functions and sum. However, this 2-Layer width-$(2n+1)$ Kolmogorov-Arnold representation may not be smooth due to its limited expressive power. We augment its expressive power by generalizing it to arbitrary depths and widths.

# ### Kolmogorov-Arnold Network (KAN)

# The Kolmogorov-Arnold representation can be written in matrix form
# 
# $$f(x)={\bf \Phi}_{\rm out}\circ{\bf \Phi}_{\rm in}\circ {\bf x}$$
# 
# where 
# 
# $${\bf \Phi}_{\rm in}= \begin{pmatrix} \phi_{1,1}(\cdot) & \cdots & \phi_{1,n}(\cdot) \\ \vdots & & \vdots \\ \phi_{2n+1,1}(\cdot) & \cdots & \phi_{2n+1,n}(\cdot) \end{pmatrix},\quad {\bf \Phi}_{\rm out}=\begin{pmatrix} \Phi_1(\cdot) & \cdots & \Phi_{2n+1}(\cdot)\end{pmatrix}$$

# We notice that both ${\bf \Phi}_{\rm in}$ and ${\bf \Phi}_{\rm out}$ are special cases of the following function matrix ${\bf \Phi}$ (with $n_{\rm in}$ inputs, and $n_{\rm out}$ outputs), we call a Kolmogorov-Arnold layer:
# 
# $${\bf \Phi}= \begin{pmatrix} \phi_{1,1}(\cdot) & \cdots & \phi_{1,n_{\rm in}}(\cdot) \\ \vdots & & \vdots \\ \phi_{n_{\rm out},1}(\cdot) & \cdots & \phi_{n_{\rm out},n_{\rm in}}(\cdot) \end{pmatrix}$$
# 
# ${\bf \Phi}_{\rm in}$ corresponds to $n_{\rm in}=n, n_{\rm out}=2n+1$, and ${\bf \Phi}_{\rm out}$ corresponds to $n_{\rm in}=2n+1, n_{\rm out}=1$.

# After defining the layer, we can construct a Kolmogorov-Arnold network simply by stacking layers! Let's say we have $L$ layers, with the $l^{\rm th}$ layer ${\bf \Phi}_l$ have shape $(n_{l+1}, n_{l})$. Then the whole network is
# 
# $${\rm KAN}({\bf x})={\bf \Phi}_{L-1}\circ\cdots \circ{\bf \Phi}_1\circ{\bf \Phi}_0\circ {\bf x}$$

# In constrast, a Multi-Layer Perceptron is interleaved by linear layers ${\bf W}_l$ and nonlinearities $\sigma$:
# 
# $${\rm MLP}({\bf x})={\bf W}_{L-1}\circ\sigma\circ\cdots\circ {\bf W}_1\circ\sigma\circ {\bf W}_0\circ {\bf x}$$

# A KAN can be easily visualized. (1) A KAN is simply stack of KAN layers. (2) Each KAN layer can be visualized as a fully-connected layer, with a 1D function placed on each edge. Let's see an example below.

# ### Get started with KANs

# Initialize KAN

# In[74]:


from kan import *
torch.set_default_dtype(torch.float64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
model = KAN(width=[2,4,1], grid=5, k=7, seed=42, device=device)


# In[ ]:



# In[75]:


from kan.utils import create_dataset
# create dataset f(x,y) = exp(sin(pi*x)+y^2)
f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)

dataset = create_dataset(f, n_var=2, device=device)

dataset['train_input'] = dataset['train_input'] # + (0.01)*torch.randn(1000,2)
dataset['test_input'] = dataset['test_input'] #+ (0.01)*torch.randn(1000,2)
dataset['train_input'].shape, dataset['train_label'].shape


# Plot KAN at initialization

# In[76]:


# plot KAN at initialization
model(dataset['train_input']);
model.plot()


# Train KAN with sparsity regularization

# In[77]:


# train the model
model.fit(dataset, opt="LBFGS", steps=50, lamb=0.001);


# Plot trained KAN

# In[78]:


model.plot()


# Prune KAN and replot

# In[79]:


model = model.prune()
model.plot()


# Continue training and replot

# In[80]:


model.fit(dataset, opt="LBFGS", steps=50);


# In[81]:


model = model.refine(10)


# In[82]:


model.fit(dataset, opt="LBFGS", steps=50);


# Automatically or manually set activation functions to be symbolic

# In[83]:


mode = "auto" # "manual"

if mode == "manual":
    # manual mode
    model.fix_symbolic(0,0,0,'sin');
    model.fix_symbolic(0,1,0,'x^2');
    model.fix_symbolic(1,0,0,'exp');
elif mode == "auto":
    # automatic mode
    lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs']
    model.auto_symbolic(lib=lib)


# Continue training till machine precision

# In[84]:


model.fit(dataset, opt="LBFGS", steps=50);


# Obtain the symbolic formula

# In[85]:


from kan.utils import ex_round

ex_round(model.symbolic_formula()[0][0],4)


# In[86]:


model(torch.tensor([[1.0],[2.0]]).T)


# In[87]:


import matplotlib.pyplot as plt
import numpy as np

# Assuming 'dataset' is already loaded and 'model' is a trained model
# dataset["train_input"] shape: (num_samples, 2) -> x and y
# dataset["train_label"] shape: (num_samples,) -> z (true labels)

everyOther = 30

# Extract x, y from dataset["train_input"] and z from dataset["train_label"]
x = dataset["train_input"][::everyOther, 0]  # First column is x
y = dataset["train_input"][::everyOther, 1]  # Second column is y
z = dataset["train_label"][::everyOther]        # True labels for z



# In[88]:


print(x)


# In[89]:


# Generate model predictions using dataset["train_input"]
model_predictions = model(dataset["train_input"].T)[::everyOther]

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the true data (x, y, z)
ax.scatter(x.detach().numpy(), y.detach().numpy(), z.detach().numpy(), color='blue', label='True Labels', alpha=0.1)

# Plot the model's predicted data on top (x, y, predicted_z)
ax.scatter(x.detach().numpy(), y.detach().numpy(), model_predictions.detach().numpy(), color='red', label='Model Predictions', alpha=0.9)

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z (Labels)')
ax.set_title('True Labels vs Model Predictions')

# Add legend
ax.legend()

# Show the plot
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




