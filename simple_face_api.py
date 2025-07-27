import boto3
import base64
import os
# Add your credentials directly
rekognition = boto3.client(
    'rekognition',
    aws_access_key_id='AKIARLPXLUEWWLRIZFW4',       
    aws_secret_access_key='6Kaiu1Y4WLsakdRRpZ7GyMSw/hHsvMmuhMI6c5g4', 
    region_name='us-east-1'                        
)

def read_image_file(image_path):
    """Read image file and convert to base64"""
    try:
        with open(image_path, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/jpeg;base64,{image_data}"
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found")
        return None
    except Exception as e:
        print(f"Error reading image: {str(e)}")
        return None

def compare_faces(image1_path, image2_path):
    """Compare faces between two images"""
    
    # Read and encode images
    image1_b64 = read_image_file(image1_path)
    image2_b64 = read_image_file(image2_path)
    
    if not image1_b64 or not image2_b64:
        return {"error": "Failed to read one or both images"}
    
    try:
        # Decode base64 images
        img1 = base64.b64decode(image1_b64.split(',')[1])
        img2 = base64.b64decode(image2_b64.split(',')[1])
        
        # Compare faces using AWS Rekognition
        result = rekognition.compare_faces(
            SourceImage={'Bytes': img1},
            TargetImage={'Bytes': img2}
        )
        
        # Return simple response
        if result['FaceMatches']:
            similarity = result['FaceMatches'][0]['Similarity']
            return {
                "status": "present", 
                "similarity": round(similarity, 2),
                "message": f"Same face detected with {similarity:.1f}% similarity"
            }
        else:
            return {
                "status": "absent", 
                "similarity": 0,
                "message": "Different faces or no face match found"
            }
            
    except Exception as e:
        return {"error": f"Face comparison failed: {str(e)}"}

def main():
    """Main function to run face comparison"""
    
    # Image file paths
    image2_path = "akarsh.jpeg"
    image1_path = "both.jpeg"
    
    print("Starting face comparison...")
    print(f"Image 1: {image1_path}")
    print(f"Image 2: {image2_path}")
    print("-" * 40)
    
    # Compare faces
    result = compare_faces(image1_path, image2_path)
    
    # Display results
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        status_emoji = "✅" if result['status'] == "present" else "❌"
        print(f"{status_emoji} Status: {result['status'].upper()}")
        print(f"📊 Similarity Score: {result['similarity']}%")
        print(f"💬 Message: {result['message']}")

if __name__ == "__main__":
    main()