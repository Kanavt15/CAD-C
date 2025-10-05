# Test Dataset Configuration for Threshold Optimization
# =====================================================

# Ground truth labels for test images
# Label: 0 = Healthy/No Cancer, 1 = Cancer/Nodule

TEST_DATASET = [
    # Main directory images
    {
        'path': 'healthy.jpg',
        'label': 0,  # Based on filename - healthy tissue
        'description': 'Healthy lung tissue - no abnormalities',
        'confidence': 'high'
    },
    {
        'path': 'download.jpg', 
        'label': 1,  # Previous inference showed cancer probability
        'description': 'Suspected nodule case',
        'confidence': 'medium'
    },
    {
        'path': 'images.jpg',
        'label': 1,  # General medical image, likely positive case
        'description': 'Medical scan - suspected cancer',
        'confidence': 'medium'
    },
    {
        'path': 'unhealty.png',
        'label': 1,  # Based on filename - unhealthy tissue
        'description': 'Unhealthy lung tissue with abnormalities',
        'confidence': 'high'
    },
    
    # Test images directory
    {
        'path': 'test_images/test_nodule_64x64.png',
        'label': 1,  # Explicitly named as nodule
        'description': 'Test nodule image 64x64 resolution',
        'confidence': 'high'
    },
    {
        'path': 'test_images/test_nodule_256x256.png',
        'label': 1,  # Explicitly named as nodule
        'description': 'Test nodule image 256x256 resolution',
        'confidence': 'high'
    }
]

# Validation notes:
# - healthy.jpg: Should have low cancer probability
# - unhealty.png and test_nodule_*.png: Should have high cancer probability
# - download.jpg and images.jpg: Medium confidence based on previous inference results