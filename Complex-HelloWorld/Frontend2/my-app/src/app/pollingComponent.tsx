"use client";
import React, { useEffect, useState } from 'react';

const PollingComponent = () => {
  const [isPolling, setIsPolling] = useState(false);
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  useEffect(() => {
    if (!isClient) return;

    const interval = setInterval(() => {
      fetch('https://a14b-198-96-33-206.ngrok-free.app/scan_results', {
        method: 'GET',
        headers: {
          'ngrok-skip-browser-warning': 'true'
        }
      })
        .then(response => {
          const contentType = response.headers.get('content-type');
          if (contentType?.includes('application/json')) {
            return response.json();
          } else {
            return response.text().then(text => {
              console.error('Invalid content type:', contentType);
              console.error('Response text:', text);
              throw new Error('Invalid content type');
            });
          }
        })
        .then(data => {
          console.log('Polling successful:', data);
          setIsPolling(true);
        })
        .catch(error => {
          console.error('Polling failed:', error);
          setIsPolling(false);
        });
    }, 3000); // Poll every 3 seconds

    return () => clearInterval(interval); // Cleanup on component unmount
  }, [isClient]);

  if (!isClient) {
    return null; // Render nothing on the server
  }

  return (
    <div>
      {isPolling ? (
        <div>Hello I am polling</div>
      ) : (
        <div>Hello I stopped polling</div>
      )}
    </div>
  );
};

export default PollingComponent;