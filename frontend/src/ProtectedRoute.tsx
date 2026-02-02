import React, { useEffect, useState } from "react";
import { Navigate, useLocation } from "react-router-dom";
import { resolveApiUrl } from "./apiBase";
import { getToken } from "./LoginPage";

interface ProtectedRouteProps {
  children: React.ReactNode;
}

export function ProtectedRoute({ children }: ProtectedRouteProps) {
  const [checked, setChecked] = useState(false);
  const [authenticated, setAuthenticated] = useState(false);
  const location = useLocation();
  const token = getToken();

  useEffect(() => {
    if (!token) {
      setAuthenticated(false);
      setChecked(true);
      return;
    }
    fetch(resolveApiUrl("/api/auth/me"), {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => {
        setAuthenticated(res.ok);
        setChecked(true);
      })
      .catch(() => {
        setAuthenticated(false);
        setChecked(true);
      });
  }, [token]);

  if (!checked) {
    return (
      <div className="login-page" style={{ alignItems: "center", justifyContent: "center" }}>
        <p className="login-subtitle">Checking authentication…</p>
      </div>
    );
  }

  if (!authenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return <>{children}</>;
}
