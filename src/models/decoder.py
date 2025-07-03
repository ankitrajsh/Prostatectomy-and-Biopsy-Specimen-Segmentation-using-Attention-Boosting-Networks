class DecoderWithAfN(nn.Module):
    def __init__(self, encoder_channels=[512, 256, 128, 64], decoder_channels=[256, 128, 64, 32]):
        super(DecoderWithAfN, self).__init__()
        self.decoder_blocks = nn.ModuleList(
            [AfN(enc_ch, dec_ch) for enc_ch, dec_ch in zip(encoder_channels, decoder_channels)]
        )
        self.final_conv = nn.Conv2d(decoder_channels[-1], 1, kernel_size=1)  # Final layer with 1 class (binary)
        self.softmax = nn.Softmax(dim=1)  # For multi-class replace with nn.Softmax(dim=1)

    def forward(self, features):
        x = features[-1]  # Start with the last feature map from DbN (smallest spatial size)
        for i in range(len(self.decoder_blocks)):
            x = self.decoder_blocks[i](features[-(i+2)], x)  # Fuse encoder and up-sampled decoder features
        x = self.final_conv(x)  # Final 1x1 convolution
        return self.softmax(x)