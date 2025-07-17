import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Plotting settings
plt.close("all")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10


# Define Neural Network Modelarchitecture
def create_model():
    model = {
        "dense1:": tf.keras.layers.Dense(50, activation="tanh"),
        "dense2:": tf.keras.layers.Dense(50, activation="tanh"),
        'dense3:':tf.keras.layers.Dense(50, activation='tanh'),
        'output:': tf.keras.layers.Dense(1, activation=None)
    }
    return model

# Define Forward Pasa
def forward_pass(model,X):
    x = model['dense1:'](X)
    x = model['dense2:'](x)
    x = model['dense3:'](x)
    x = model['output:'](x)
    return x

model = create_model()
print("\nModel created successfully.\n")
print(model)
    

# Partial Differential Equation (PDE) Loss Function
def pde(x,model):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y_pred = forward_pass(model, x)
        y_x= tape.gradient(y_pred, x)
    
    y_xx = tape.gradient(y_x, x)
    del tape  # Free memory
    return y_xx + np.pi**2 * tf.sin(np.pi * x)


# Loss Function for Boundary Conditions
def loss(model , x , x_bc , y_bc):
    res = pde(x,model)
    loss_pde = tf.reduce_mean(tf.square(res))
    # loss_pde = How badly is the model violating the PDE equation at interior points?
    y_pred_bc = forward_pass(model, x_bc)
    loss_bc = tf.reduce_mean(tf.square(y_pred_bc - y_bc))
    # loss_bc = How badly is the model violating the boundary conditions?
    return loss_pde + loss_bc


# Training Function
def train(model, x, x_bc, y_bc, optimizer):
    with tf.GradientTape() as tape:
        loss_value = loss(model, x, x_bc, y_bc)

    variables = [var for layer in model.values() for var in layer.trainable_variables]
    grads = tape.gradient(loss_value, variables)
    
    optimizer.apply_gradients(zip(grads, variables))
    return loss_value


x_train = np.linspace(-1,1,100).reshape(-1,1)
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)

# Boundy Data
x_bc = np.array([[-1.0], [1.0]], dtype=np.float32)
y_bc = np.array([[0.0], [0.0]], dtype=np.float32)
x_bc = tf.convert_to_tensor(x_bc, dtype=tf.float32)
y_bc = tf.convert_to_tensor(y_bc, dtype=tf.float32)

# model
model = create_model()

# define optimizer
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=1000,
    decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Train
epochs = 10000
for epoch in range(epochs):
    loss_value = train(model, x_train, x_bc, y_bc, optimizer)
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss_value.numpy()}")
        
        
# Prediction
x_test = np.linspace(-1, 1, 1000).reshape(-1,1)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
y_pred = forward_pass(model, x_test).numpy()

y_true = np.sin(np.pi * x_test)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(x_test, y_pred, label='PINN Solution', color='blue')
plt.plot(x_test, y_true, label='Analytical Solution', color='red', linestyle='--')
plt.title('Solution of the Poisson Equation')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
print("Training completed and results plotted.")
    
        