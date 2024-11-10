"use client";
import React, { useEffect, useState } from 'react';

const PollingComponent = () => {
  const [isPolling, setIsPolling] = useState(false);
  const [isClient, setIsClient] = useState(false);
  const [data, setData] = useState([]);

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
          setData(data);
          setIsPolling(true);

          // Check the length of the data and send a POST request if greater than 3
          if (data.length > 3) {
            fetch('http://localhost:3001/finalCall/run-script/', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json'
              },
              body: JSON.stringify({ message: 'Data length is greater than 3' })
            })
              .then(response => response.json())
              .then(postData => {
                console.log('POST successful:', postData);
              })
              .catch(postError => {
                console.error('POST failed:', postError);
              });
          }
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
        <div>
          <div>Hello I am polling</div>
          <pre>{JSON.stringify(data, null, 2)}</pre>
        </div>
      ) : (
        <div>Hello I stopped polling</div>
      )}
    </div>
  );
};

export default PollingComponent;