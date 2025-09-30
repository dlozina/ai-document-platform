"""
MinIO Object Storage integration for Ingestion Service
"""

import logging
from typing import Optional, Dict, Any, BinaryIO, List
from minio import Minio
from minio.error import S3Error
import io

from .config import get_settings

logger = logging.getLogger(__name__)


class MinIOManager:
    """MinIO object storage manager."""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = Minio(
            self.settings.minio_endpoint,
            access_key=self.settings.minio_access_key,
            secret_key=self.settings.minio_secret_key,
            secure=self.settings.minio_secure
        )
        self.bucket_prefix = self.settings.minio_bucket_prefix
    
    def ensure_bucket_exists(self, bucket_name: str) -> bool:
        """
        Ensure bucket exists, create if it doesn't.
        
        Args:
            bucket_name: Name of the bucket
            
        Returns:
            True if bucket exists or was created successfully
        """
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                logger.info(f"Created bucket: {bucket_name}")
            return True
        except S3Error as e:
            logger.error(f"Failed to create bucket {bucket_name}: {e}")
            return False
    
    def get_tenant_bucket_name(self, tenant_id: str) -> str:
        """
        Get bucket name for tenant.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            Bucket name
        """
        return f"{self.bucket_prefix}-{tenant_id}"
    
    def upload_file(
        self, 
        tenant_id: str, 
        file_path: str, 
        file_content: bytes,
        content_type: str
    ) -> bool:
        """
        Upload file to MinIO.
        
        Args:
            tenant_id: Tenant identifier
            file_path: Path where file should be stored
            file_content: File content as bytes
            content_type: MIME content type
            
        Returns:
            True if upload successful
        """
        try:
            bucket_name = self.get_tenant_bucket_name(tenant_id)
            self.ensure_bucket_exists(bucket_name)
            
            # Upload file
            file_obj = io.BytesIO(file_content)
            self.client.put_object(
                bucket_name,
                file_path,
                file_obj,
                length=len(file_content),
                content_type=content_type
            )
            
            logger.info(f"Uploaded file {file_path} to bucket {bucket_name}")
            return True
            
        except S3Error as e:
            logger.error(f"Failed to upload file {file_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error uploading file {file_path}: {e}")
            return False
    
    def download_file(self, tenant_id: str, file_path: str) -> Optional[bytes]:
        """
        Download file from MinIO.
        
        Args:
            tenant_id: Tenant identifier
            file_path: Path of the file to download
            
        Returns:
            File content as bytes, or None if failed
        """
        try:
            bucket_name = self.get_tenant_bucket_name(tenant_id)
            
            response = self.client.get_object(bucket_name, file_path)
            file_content = response.read()
            response.close()
            response.release_conn()
            
            logger.info(f"Downloaded file {file_path} from bucket {bucket_name}")
            return file_content
            
        except S3Error as e:
            logger.error(f"Failed to download file {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error downloading file {file_path}: {e}")
            return None
    
    def delete_file(self, tenant_id: str, file_path: str) -> bool:
        """
        Delete file from MinIO.
        
        Args:
            tenant_id: Tenant identifier
            file_path: Path of the file to delete
            
        Returns:
            True if deletion successful
        """
        try:
            bucket_name = self.get_tenant_bucket_name(tenant_id)
            self.client.remove_object(bucket_name, file_path)
            
            logger.info(f"Deleted file {file_path} from bucket {bucket_name}")
            return True
            
        except S3Error as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting file {file_path}: {e}")
            return False
    
    def file_exists(self, tenant_id: str, file_path: str) -> bool:
        """
        Check if file exists in MinIO.
        
        Args:
            tenant_id: Tenant identifier
            file_path: Path of the file to check
            
        Returns:
            True if file exists
        """
        try:
            bucket_name = self.get_tenant_bucket_name(tenant_id)
            self.client.stat_object(bucket_name, file_path)
            return True
        except S3Error:
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking file {file_path}: {e}")
            return False
    
    def get_file_info(self, tenant_id: str, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get file information from MinIO.
        
        Args:
            tenant_id: Tenant identifier
            file_path: Path of the file
            
        Returns:
            File information dictionary, or None if failed
        """
        try:
            bucket_name = self.get_tenant_bucket_name(tenant_id)
            stat = self.client.stat_object(bucket_name, file_path)
            
            return {
                "size": stat.size,
                "etag": stat.etag,
                "last_modified": stat.last_modified,
                "content_type": stat.content_type,
                "metadata": stat.metadata
            }
            
        except S3Error as e:
            logger.error(f"Failed to get file info for {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting file info for {file_path}: {e}")
            return None
    
    def list_files(
        self, 
        tenant_id: str, 
        prefix: str = "", 
        recursive: bool = True
    ) -> List[Dict[str, Any]]:
        """
        List files in tenant bucket.
        
        Args:
            tenant_id: Tenant identifier
            prefix: File path prefix to filter
            recursive: Whether to list recursively
            
        Returns:
            List of file information dictionaries
        """
        try:
            bucket_name = self.get_tenant_bucket_name(tenant_id)
            objects = self.client.list_objects(
                bucket_name, 
                prefix=prefix, 
                recursive=recursive
            )
            
            files = []
            for obj in objects:
                files.append({
                    "object_name": obj.object_name,
                    "size": obj.size,
                    "etag": obj.etag,
                    "last_modified": obj.last_modified,
                    "content_type": obj.content_type
                })
            
            return files
            
        except S3Error as e:
            logger.error(f"Failed to list files in bucket {bucket_name}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error listing files: {e}")
            return []
    
    def get_bucket_size(self, tenant_id: str) -> int:
        """
        Get total size of files in tenant bucket.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            Total size in bytes
        """
        try:
            bucket_name = self.get_tenant_bucket_name(tenant_id)
            objects = self.client.list_objects(bucket_name, recursive=True)
            
            total_size = 0
            for obj in objects:
                total_size += obj.size
            
            return total_size
            
        except S3Error as e:
            logger.error(f"Failed to get bucket size for {bucket_name}: {e}")
            return 0
        except Exception as e:
            logger.error(f"Unexpected error getting bucket size: {e}")
            return 0
    
    def test_connection(self) -> bool:
        """
        Test MinIO connection.
        
        Returns:
            True if connection is successful
        """
        try:
            # Try to list buckets
            buckets = self.client.list_buckets()
            logger.info("MinIO connection test successful")
            return True
        except Exception as e:
            logger.error(f"MinIO connection test failed: {e}")
            return False


# Global MinIO manager instance
minio_manager: Optional[MinIOManager] = None


def get_minio_manager() -> MinIOManager:
    """Get MinIO manager instance."""
    global minio_manager
    if minio_manager is None:
        minio_manager = MinIOManager()
    return minio_manager
