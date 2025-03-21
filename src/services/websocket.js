// src/services/websocket.js
import io from "socket.io-client";

const socket = io("http://localhost:3001");

export const subscribeToUpdates = (callback) => {
  socket.on("data-update", (data) => {
    callback(data);
  });
};

export const unsubscribeFromUpdates = () => {
  socket.off("data-update");
};
