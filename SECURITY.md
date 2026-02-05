# Security Policy

## Supported Versions

We currently support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.9.x   | :white_check_mark: |
| 0.5.x   | :white_check_mark: |
| < 0.5   | :x:                |

## Reporting a Vulnerability

**Please DO NOT report security vulnerabilities through public GitHub issues.**

### For Security Issues

If you discover a security vulnerability, please email the maintainers directly:

- **DO NOT** create a public GitHub issue
- **DO NOT** discuss the vulnerability publicly until it's been addressed

When reporting, please include:

1. **Description** of the vulnerability
2. **Steps to reproduce** the issue
3. **Potential impact** assessment
4. **Suggested fix** (if available)

We will acknowledge receipt within 48 hours and provide a detailed response within 5 business days.

## Security Considerations for Medical AI

### Data Privacy & HIPAA Compliance

This software handles sensitive medical data. Users must:

1. **De-identify all data** before processing
2. **Encrypt data** at rest and in transit
3. **Control access** with proper authentication
4. **Audit all operations** with comprehensive logging
5. **Follow institutional policies** for data handling

See [docs/ETHICS.md](docs/ETHICS.md) for detailed HIPAA compliance guidelines.

### Model Security

#### Input Validation
- All image inputs are validated for format and size
- Malformed inputs are rejected with appropriate error messages
- No user input is directly executed as code

#### API Security
When deploying the API:

```python
# ALWAYS use authentication in production
from functools import wraps
from flask import request, jsonify
import os

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != os.environ.get('API_KEY'):
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function
```

#### Model Poisoning
- Only load models from trusted sources
- Verify model checksums before loading
- Never load arbitrary model files from user input

### Deployment Security

#### Docker Security
- Use minimal base images
- Run containers as non-root user
- Scan images for vulnerabilities regularly
- Keep base images updated

```dockerfile
# Good practice - run as non-root
FROM python:3.9-slim
RUN useradd -m -u 1000 appuser
USER appuser
```

#### Environment Variables
- NEVER commit secrets to git
- Use environment variables for sensitive config
- Use secret management services (AWS Secrets Manager, etc.)

```python
# Good practice - use environment variables
import os
from dotenv import load_env()

load_dotenv()
API_KEY = os.getenv('API_KEY')
DATABASE_URL = os.getenv('DATABASE_URL')
```

#### Network Security
- Use HTTPS/TLS for all communications
- Implement rate limiting on API endpoints
- Use firewall rules to restrict access
- Monitor for unusual traffic patterns

### Dependency Security

We regularly check dependencies for known vulnerabilities:

```bash
# Check for vulnerabilities
pip install safety
safety check

# Update dependencies
pip install --upgrade -r requirements.txt
```

### Code Security

#### Avoid Code Injection
```python
# BAD - Never do this
eval(user_input)
exec(user_input)
os.system(user_input)

# GOOD - Validate and sanitize
if user_input in ALLOWED_OPTIONS:
    process(user_input)
```

#### Secure File Handling
```python
# BAD - Path traversal vulnerability
filepath = os.path.join(base_dir, user_filename)

# GOOD - Validate paths
from pathlib import Path
filepath = Path(base_dir) / user_filename
if not filepath.resolve().is_relative_to(Path(base_dir).resolve()):
    raise ValueError("Invalid file path")
```

## Known Security Considerations

### Medical Image Format Vulnerabilities

**DICOM Files**:
- DICOM format can contain executable code
- Always validate and sanitize DICOM files
- Use trusted parsing libraries (pydicom)
- Never execute embedded code from DICOM files

**NIfTI Files**:
- NIfTI files can contain large arrays that cause memory issues
- Implement file size limits
- Use streaming processing for large files

### Model Inference Attacks

**Adversarial Examples**:
- Medical images can be subtly modified to fool models
- Implement input validation and anomaly detection
- Use uncertainty quantification to flag suspicious inputs
- See `src/evaluation/uncertainty.py` for tools

**Model Inversion**:
- Attackers may try to extract training data from model
- Use differential privacy techniques if needed
- Limit information in error messages
- Monitor for systematic probing

## Security Best Practices

### For Developers

1. **Code Review**: All code changes require review
2. **Static Analysis**: Use bandit, flake8 for security checks
3. **Dependency Scanning**: Regularly check for vulnerable dependencies
4. **Secrets Management**: Never commit secrets, use environment variables
5. **Input Validation**: Validate all user inputs
6. **Error Handling**: Don't expose stack traces to users
7. **Logging**: Log security events, but not sensitive data

### For Deployers

1. **Authentication**: Always use authentication in production
2. **Authorization**: Implement role-based access control
3. **Encryption**: Use TLS/SSL for all communications
4. **Monitoring**: Monitor for security events and anomalies
5. **Updates**: Keep all dependencies updated
6. **Backups**: Regular backups with encryption
7. **Incident Response**: Have a plan for security incidents

### For Users

1. **Data Protection**: Ensure data is de-identified and encrypted
2. **Access Control**: Limit access to authorized personnel only
3. **Compliance**: Follow HIPAA and institutional policies
4. **Validation**: Validate model outputs before clinical use
5. **Updates**: Keep software updated with security patches
6. **Training**: Train staff on security best practices
7. **Auditing**: Regular security audits and assessments

## Regulatory Compliance

### HIPAA Compliance Checklist

- [ ] **Data De-identification**: All PII removed from images
- [ ] **Encryption**: Data encrypted at rest and in transit
- [ ] **Access Controls**: Strong authentication and authorization
- [ ] **Audit Logs**: Comprehensive logging of all access
- [ ] **Business Associate Agreement**: In place if applicable
- [ ] **Security Risk Assessment**: Completed annually
- [ ] **Incident Response Plan**: Documented and tested
- [ ] **Staff Training**: HIPAA training for all users

### GDPR Compliance (if applicable)

- [ ] **Data Minimization**: Only collect necessary data
- [ ] **Purpose Limitation**: Clear purpose for data use
- [ ] **Consent**: Proper consent mechanisms
- [ ] **Right to Erasure**: Ability to delete user data
- [ ] **Data Portability**: Ability to export data
- [ ] **Privacy by Design**: Security built into system

## Vulnerability Disclosure Timeline

1. **Day 0**: Vulnerability reported
2. **Day 2**: Acknowledgment sent to reporter
3. **Day 5**: Initial assessment and response
4. **Day 30**: Fix developed and tested
5. **Day 35**: Security patch released
6. **Day 40**: Public disclosure (coordinated with reporter)

## Security Updates

Security updates are released as patch versions:
- `0.9.x` - Latest stable with security patches
- Security advisories posted to GitHub Security tab
- Critical issues announced via repository

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [HIPAA Security Rule](https://www.hhs.gov/hipaa/for-professionals/security/index.html)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [CWE/SANS Top 25](https://cwe.mitre.org/top25/)

## Contact

For security concerns:
- **Email**: [Repository maintainers - do not create public issues]
- **PGP Key**: [If available]

---

**Remember**: Security is everyone's responsibility. When in doubt, ask before proceeding.
