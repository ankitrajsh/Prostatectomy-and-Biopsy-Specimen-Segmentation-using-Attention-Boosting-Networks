class AbM(nn.Module):
    def __init__(self, in_channels):
        super(AbM, self).__init__()
        self.spatial_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.channel_att = AbG(in_channels)  # Using AbG for channel attention

    def forward(self, x):
        spatial_out = self.spatial_conv(x)
        channel_out = self.channel_att(spatial_out)
        return channel_out + spatial_out