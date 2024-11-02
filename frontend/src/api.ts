import axios from "axios";

export const preprocess = async (file: File): Promise<string> => {
  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await axios.post(
      "http://127.0.0.1:5000/api/preprocess",
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      }
    );

    const { png_data } = response.data;

    // downloadFile(png_data, "image/png", "output.png");
    // downloadFile(svg_data, "image/svg+xml", "output.svg");

    const pngUrl = createObjectUrl(png_data, "image/png");

    return pngUrl;
  } catch (error) {
    console.error("Error posting data:", error);
    return "";
  }
};

export const process = async (border: {
  left: number;
  top: number;
  width: number;
  height: number;
}): Promise<{ pngUrl: string; svgUrl: string }> => {
  try {
    const response = await axios.post(
      "http://127.0.0.1:5000/api/process",
      border
    );

    const { png_data, svg_data } = response.data;

    // downloadFile(png_data, "image/png", "output.png");
    // downloadFile(svg_data, "image/svg+xml", "output.svg");

    const pngUrl = createObjectUrl(png_data, "image/png");
    const svgUrl = createObjectUrl(svg_data, "image/svg+xml");

    return { pngUrl, svgUrl };
  } catch (error) {
    console.error("Error posting data:", error);
    return { pngUrl: "", svgUrl: "" };
  }
};

function createObjectUrl(content: string, type: string) {
  const blobPart = base64ToArrayBuffer(content);
  const blob = new Blob([blobPart], { type });
  return window.URL.createObjectURL(blob);
}

function downloadFile(content: string, type: string, name: string) {
  const blobPart = base64ToArrayBuffer(content);
  const blob = new Blob([blobPart], { type });
  const link = document.createElement("a");
  link.href = window.URL.createObjectURL(blob);
  link.download = name;
  link.click();
}

export const base64ToArrayBuffer = (base64: any) => {
  const binaryString = window.atob(base64);
  const binaryLen = binaryString.length;
  const bytes = new Uint8Array(binaryLen);
  for (let i = 0; i < binaryLen; i++) {
    const ascii = binaryString.charCodeAt(i);
    bytes[i] = ascii;
  }
  return bytes;
};

export interface ReturnResult {
  fileContent: string;
  fileName: string;
  fileType: string;
}
