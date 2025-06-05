#!/bin/bash
# Deployment script for Raspberry Pi

# Configuration
PI_HOST="raspberrypi.local"
PI_USER="pi"
PROJECT_DIR="/home/pi/cutlery-classifier"
MODEL_PATH="models/exports/cutlery_classifier_edge.onnx"
SERVICE_NAME="cutlery_classifier"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_error() {
    if [ $? -ne 0 ]; then
        log_error "$1"
        exit 1
    fi
}

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    log_error "Model not found at $MODEL_PATH"
    log_info "Run 'python scripts/export_model.py --target pi' first"
    exit 1
fi

# Export model for Raspberry Pi
log_info "Exporting model for Raspberry Pi..."
python scripts/export_model.py --target pi --verify
check_error "Model export failed"

# Create deployment package
log_info "Creating deployment package..."
DEPLOY_DIR="deploy_pkg"
rm -rf $DEPLOY_DIR
mkdir -p $DEPLOY_DIR

# Copy required files
cp -r scripts/ $DEPLOY_DIR/
cp -r src/ $DEPLOY_DIR/
cp -r models/exports/ $DEPLOY_DIR/models/
cp requirements.txt $DEPLOY_DIR/
cp docs/raspberry_pi_guide.md $DEPLOY_DIR/

# Create systemd service file
cat > $DEPLOY_DIR/cutlery_classifier.service << EOL
[Unit]
Description=Cutlery Classifier Service
After=network.target

[Service]
Type=simple
User=$PI_USER
WorkingDirectory=$PROJECT_DIR
ExecStart=/usr/bin/python3 scripts/run_inference_on_pi.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOL

# Create health check script
cat > $DEPLOY_DIR/monitor_health.sh << EOL
#!/bin/bash

# Check if service is running
if ! systemctl is-active --quiet $SERVICE_NAME; then
    systemctl restart $SERVICE_NAME
    echo "\$(date): Service restarted due to failure" >> /var/log/cutlery_classifier.log
fi

# Check memory usage
MEM_FREE=\$(free -m | awk '/^Mem:/{print \$4}')
if [ \$MEM_FREE -lt 100 ]; then
    echo "\$(date): Low memory warning (\${MEM_FREE}MB free)" >> /var/log/cutlery_classifier.log
fi

# Check temperature
TEMP=\$(vcgencmd measure_temp | cut -d= -f2 | cut -d"'" -f1)
if [ \$(echo "\$TEMP > 80" | bc) -eq 1 ]; then
    echo "\$(date): High temperature warning (\${TEMP}°C)" >> /var/log/cutlery_classifier.log
fi
EOL

chmod +x $DEPLOY_DIR/monitor_health.sh

# Create setup script
cat > $DEPLOY_DIR/setup.sh << EOL
#!/bin/bash

# Install dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-opencv
pip3 install -r requirements.txt

# Setup service
sudo cp cutlery_classifier.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable cutlery_classifier
sudo systemctl start cutlery_classifier

# Setup health monitoring
sudo cp monitor_health.sh /usr/local/bin/
(crontab -l 2>/dev/null; echo "*/5 * * * * /usr/local/bin/monitor_health.sh") | crontab -

# Setup logging
sudo touch /var/log/cutlery_classifier.log
sudo chown $PI_USER:$PI_USER /var/log/cutlery_classifier.log

# Optimize system settings
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
EOL

chmod +x $DEPLOY_DIR/setup.sh

# Create deployment package
log_info "Creating deployment archive..."
tar czf deploy.tar.gz -C $DEPLOY_DIR .
check_error "Failed to create deployment archive"

# Deploy to Raspberry Pi
log_info "Deploying to Raspberry Pi..."
ssh $PI_USER@$PI_HOST "mkdir -p $PROJECT_DIR"
check_error "Failed to create project directory on Raspberry Pi"

scp deploy.tar.gz $PI_USER@$PI_HOST:$PROJECT_DIR/
check_error "Failed to copy deployment package"

# Execute setup on Raspberry Pi
ssh $PI_USER@$PI_HOST "cd $PROJECT_DIR && \
    tar xzf deploy.tar.gz && \
    rm deploy.tar.gz && \
    ./setup.sh"
check_error "Setup failed on Raspberry Pi"

# Cleanup
rm -rf $DEPLOY_DIR deploy.tar.gz

# Verify deployment
log_info "Verifying deployment..."
ssh $PI_USER@$PI_HOST "systemctl is-active $SERVICE_NAME"
check_error "Service verification failed"

log_info "Deployment completed successfully!"
log_info "Monitor the service with: systemctl status $SERVICE_NAME"
log_info "View logs with: tail -f /var/log/cutlery_classifier.log" 