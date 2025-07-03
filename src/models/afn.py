class AfN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AfN, self).__init__()
        self.attention = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, encoder_feature, decoder_feature):
        encoder_attention = self.attention(encoder_feature)
        upsampled = self.upconv(decoder_feature)
        fused = self.relu(encoder_attention + upsampled)
        return fused