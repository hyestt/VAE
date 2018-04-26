# VAE


Function in VAE

1) generate samples(self, z np): Generate random samples from the provided latent samples.

2) encoder(self, x): Build a two-layer network. Two output branches for the two ( nlatent)-dim vectors representing the z mean and z log var.

3) decoder(self, z): Build a two-layer network. Use two fully-connected layers with 50 nodes and then 100 nodes. The output is a tensor containing the decoded images.

4) latent loss(self, z mean, z log var): Calculate the latent loss.

5) reconstruction loss(self, f, x gt): Calculate the reconstruction loss.

6) loss(self, f, x gt, z mean, z var): The total loss of your VAE. 

7) update op(self, loss, learning rate): Perform parameter update. Use tf.train.AdamOptimizer to minimize the loss of VAE and update the parameters.

The result graph: 
