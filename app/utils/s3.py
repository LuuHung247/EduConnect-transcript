import os
import boto3
from botocore.exceptions import ClientError
from typing import Optional

def _get_s3_client():
    """Get configured S3 client"""
    try:
        return boto3.client(
            's3',
            region_name=os.environ.get('AWS_REGION', 'ap-southeast-1'),
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
        )
    except Exception as e:
        print(f"Failed to create S3 client: {e}")
        return None


def upload_to_s3(buffer, key: str, content_type: str, prefix: str = "") -> str:
    """
    Upload bytes buffer to S3 and return public URL
    
    Args:
        buffer: File content as bytes
        key: File name/key
        content_type: MIME type (e.g., 'image/jpeg', 'video/mp4')
        prefix: Optional folder prefix (e.g., 'files/user-123/thumbnail')
    
    Returns:
        Public URL of uploaded file
    
    Environment variables:
        - S3_BUCKET_NAME: Required - S3 bucket name
        - AWS_REGION: Optional - AWS region (default: ap-southeast-1)
        - AWS_ACCESS_KEY_ID: Optional - AWS access key
        - AWS_SECRET_ACCESS_KEY: Optional - AWS secret key
    """
    bucket = os.environ.get('AWS_S3_BUCKET')
    region = os.environ.get('AWS_REGION', 'ap-southeast-1')
    
    if not bucket:
        raise ValueError("AWS_S3_BUCKET environment variable not set")

    s3 = _get_s3_client()
    if not s3:
        raise RuntimeError("Failed to create S3 client")
    
    # Build full key with prefix
    full_key = f"{prefix}/{key}" if prefix else key
    
    try:
        # Upload to S3
        s3.put_object(
            Bucket=bucket,
            Key=full_key,
            Body=buffer,
            ContentType=content_type,
            # Make it publicly readable (optional)
            ACL='public-read'
        )
        
        # Return S3 URL
        return f"https://{bucket}.s3.{region}.amazonaws.com/{full_key}"
        
    except ClientError as e:
        raise RuntimeError(f"Failed to upload to S3: {e}")


def delete_from_s3(url_or_key: str) -> bool:
    """
    Delete object from S3
    
    Args:
        url_or_key: Either full S3 URL or object key
    
    Returns:
        True if deleted successfully, False otherwise
    """
    bucket = os.environ.get('AWS_S3_BUCKET')

    if not bucket:
        print("AWS_S3_BUCKET not configured, skipping delete")
        return False
    
    s3 = _get_s3_client()
    if not s3:
        print("Failed to create S3 client")
        return False
    
    # Parse key from URL if needed
    if url_or_key.startswith('http://') or url_or_key.startswith('https://'):
        # Extract key from URL
        # Example: https://bucket.s3.region.amazonaws.com/path/to/file.jpg
        parts = url_or_key.split('/')
        if len(parts) >= 4:
            key = '/'.join(parts[3:])
        else:
            print(f"Invalid S3 URL format: {url_or_key}")
            return False
    else:
        key = url_or_key
    
    try:
        s3.delete_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        print(f"Failed to delete from S3: {e}")
        return False


def download_from_s3(key: str, local_path: str) -> bool:
    """
    Download object from S3 to local file

    Args:
        key: Object key in S3
        local_path: Local file path to save downloaded content

    Returns:
        True if downloaded successfully, False otherwise
    """
    bucket = os.environ.get('AWS_S3_BUCKET')

    if not bucket:
        print("AWS_S3_BUCKET not configured")
        return False

    s3 = _get_s3_client()
    if not s3:
        return False

    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Download file
        s3.download_file(bucket, key, local_path)
        return True
    except ClientError as e:
        print(f"Failed to download from S3: {e}")
        return False


def generate_presigned_url(key: str, expiration: int = 3600) -> Optional[str]:
    """
    Generate presigned URL for temporary access to private S3 object

    Args:
        key: Object key in S3
        expiration: URL expiration time in seconds (default: 1 hour)

    Returns:
        Presigned URL or None if failed
    """
    bucket = os.environ.get('AWS_S3_BUCKET')

    if not bucket:
        print("AWS_S3_BUCKET not configured")
        return None

    s3 = _get_s3_client()
    if not s3:
        return None

    try:
        url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': key},
            ExpiresIn=expiration
        )
        return url
    except ClientError as e:
        print(f"Failed to generate presigned URL: {e}")
        return None


# Backward compatibility - keep old function names
