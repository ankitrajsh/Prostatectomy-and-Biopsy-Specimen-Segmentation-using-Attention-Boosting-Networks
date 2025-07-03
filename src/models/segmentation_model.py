class SemanticSegmentationModel(nn.Module):
    def __init__(self, encoder_type='resnet50', pretrained=True, num_classes=21):
        super(SemanticSegmentationModel, self).__init__()
        self.encoder_with_dbn = EncoderWithDbN(base_model=encoder_type, pretrained=pretrained)
        self.decoder = DecoderWithAfN(encoder_out_channels=self.encoder_with_dbn.out_channels, num_classes=num_classes)

    def forward(self, x):
        encoder_features = self.encoder_with_dbn(x)
        output = self.decoder(encoder_features)
        return output