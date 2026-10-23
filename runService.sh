#cat  /etc/systemd/system/gutbiome.service
# nano /etc/systemd/system/gutbiome.service
# sudo systemctl stop gutbiome
# sudo systemctl start gutbiome
sudo systemctl daemon-reload
sudo systemctl enable --now gutbiome
sudo systemctl status gutbiome
